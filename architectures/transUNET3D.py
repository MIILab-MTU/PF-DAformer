import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint

import torch
import math
import copy
from torch import nn
from einops import rearrange
from functools import partial
from torch.autograd import Function



def build_transunet3d_model():
    model = TransUNet3D(
        in_channels=1,
        n_classes=1,
        base_channels=16,
        img_size=192,
        patch_size=8,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        mlp_dim=2048,
        use_domain_adapt = True
    )
    return model


#######################################Grdaient Layer@############################

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd=1.0):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

######################################################################################

class DomainClassifier(nn.Module):
    def __init__(self, embed_dims_last):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),               # [B, C, 1, 1, 1]
            nn.Flatten(),                          # [B, C]
            nn.LayerNorm(embed_dims_last),         # Safe for batch size 1
            nn.Dropout(0.2),              # ✅ First Dropout
            nn.Linear(embed_dims_last, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),              # ✅ Second Dropout
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),              # ✅ Third Dropout
            nn.Linear(64, 2)                        # logits for 2 domains
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        logits = self.classifier(x)  # [batch_size, 2]
        # if self.training:            # Only print probs during training
        #     with torch.no_grad():
        #         probs = torch.softmax(logits, dim=1)
        #         print(f"Domain probs: {probs[:4]}")
        return logits



# -------------------------
# ViT Transformer Components
# -------------------------

class Attention(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B, heads, N, head_dim
        k = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # B, heads, N, N
        attn_probs = attn_scores.softmax(dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = attn_probs @ v  # B, heads, N, head_dim
        context = context.transpose(1, 2).reshape(B, N, C)

        out = self.proj(context)
        out = self.proj_dropout(out)
        return out

class Mlp(nn.Module):
    def __init__(self, hidden_size=512, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = Mlp(hidden_size, mlp_dim, dropout)

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + h
        return x

# --------------------------
# ViT Encoder for 3D volume
# --------------------------

class ViT3DEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_size=512, patch_size=8, img_size=192,
                 num_layers=6, num_heads=8, mlp_dim=2048, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 3
        self.hidden_size = hidden_size

        # Patch embedding conv3d
        self.patch_embed = nn.Conv3d(in_channels, hidden_size,
                                     kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        self.dropout = nn.Dropout(dropout)

        # Transformer layers with gradient checkpointing support
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x):
        # x shape: (B, C=1, D=192, H=192, W=192)
        B = x.shape[0]

        x = self.patch_embed(x)  # (B, hidden_size, D/patch, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, hidden_size)

        x = x + self.position_embeddings
        x = self.dropout(x)

        # Apply transformer layers with checkpointing to save memory
        for layer in self.layers:
            x = checkpoint(layer, x)

        x = self.norm(x)  # (B, N, hidden_size)

        # Reshape back to (B, hidden_size, D/patch, H/patch, W/patch)
        dim = int(round(self.num_patches ** (1 / 3)))
        x = x.transpose(1, 2).contiguous().view(B, self.hidden_size, dim, dim, dim)
        return x

# ---------------------------
# UNet decoder blocks 3D
# ---------------------------

class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x, skip):
        # Instead of fixed scale_factor, use size of skip explicitly
        x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# -------------------------
# Encoder Conv blocks (before ViT)
# -------------------------

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

# -------------------------
# Full TransUNet3D model
# -------------------------

class TransUNet3D(nn.Module):
    def __init__(self, in_channels=1, n_classes=1, base_channels=32, img_size=192,
                 patch_size=8, hidden_size=512, num_layers=6, num_heads=8, mlp_dim=2048,use_domain_adapt=True):
        super().__init__()
        
        self.use_domain_adapt = use_domain_adapt
        if self.use_domain_adapt:
            self.domain_classifier = DomainClassifier(embed_dims_last=hidden_size)  # hidden_size=512 in your case

        # Encoder conv blocks before transformer for skip connections
        self.enc1 = ConvBlock3D(in_channels, base_channels)         # 192³ -> 192³
        self.pool1 = nn.MaxPool3d(2)                               # 192 -> 96
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)  # 96³
        self.pool2 = nn.MaxPool3d(2)                               # 96 -> 48
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4)  # 48³
        self.pool3 = nn.MaxPool3d(2)                               # 48 -> 24

        # ViT encoder on downsampled volume (24³)
        self.vit = ViT3DEncoder(in_channels=base_channels * 4,
                                hidden_size=hidden_size,
                                patch_size=patch_size,
                                img_size=img_size // 8,  # 192/8=24
                                num_layers=num_layers,
                                num_heads=num_heads,
                                mlp_dim=mlp_dim)

        # Decoder blocks
        self.dec3 = DecoderBlock3D(hidden_size, base_channels * 4, base_channels * 4)  # 24 -> 48
        self.dec2 = DecoderBlock3D(base_channels * 4, base_channels * 2, base_channels * 2)  # 48 -> 96
        self.dec1 = DecoderBlock3D(base_channels * 2, base_channels, base_channels)  # 96 -> 192

        self.final_conv = nn.Conv3d(base_channels, n_classes, kernel_size=1)

    def forward(self, x, return_features=False,use_domain_adapt=False):
        if x.ndim == 4:
            x = x.unsqueeze(1)  # (B, D, H, W) → (B, 1, D, H, W)
        # Encoder path (conv blocks + pooling)
        enc1 = self.enc1(x)      # B, 32, 192,192,192
        p1 = self.pool1(enc1)   # B, 32, 96,96,96

        enc2 = self.enc2(p1)    # B, 64, 96,96,96
        p2 = self.pool2(enc2)   # B, 64, 48,48,48

        enc3 = self.enc3(p2)    # B, 128, 48,48,48
        p3 = self.pool3(enc3)   # B, 128, 24,24,24

        # ViT encoder on lowest resolution features
        vit_out = self.vit(p3)  # B, hidden_size=512, 24,24,24

        # Decoder with skip connections
        d3 = self.dec3(vit_out, enc3)  # B, 128, 48,48,48
        d2 = self.dec2(d3, enc2)       # B, 64, 96,96,96
        d1 = self.dec1(d2, enc1)       # B, 32, 192,192,192

        out = self.final_conv(d1)

        if return_features:
            if self.use_domain_adapt:
                domain_feat = grad_reverse(vit_out)
                domain_pred = self.domain_classifier(domain_feat)
                return out, vit_out, domain_pred
            else:
                return out, vit_out
        return out
# -------------------------
# Helper functions
# -------------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -------------------------
# Test run and print info
# -------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransUNet3D().to(device)

    print(f"Total trainable parameters: {count_parameters(model)/1e6:.2f} M")

    x = torch.randn(1,1,192,192,192).to(device)

    with torch.no_grad():
        out = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")  
