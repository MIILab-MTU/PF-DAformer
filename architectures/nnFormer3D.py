import torch
import torch.nn.functional as F
from timm.layers import DropPath, to_3tuple, trunc_normal_
from torch import nn
from torch.autograd import Function

#######################################Grdaient Layer############################

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

#######################################Domain Adaptaion#############################################

class DomainClassifier(nn.Module):
    def __init__(self, embed_dims_last):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),          # [batch, embed_dims[-1], 1, 1, 1]
            nn.Flatten(),                     # [batch, embed_dims[-1]]
            nn.LayerNorm(embed_dims_last),  # Replace LayerNorm with BatchNorm
            nn.Linear(embed_dims_last, 100),
            nn.ReLU(),
            nn.Linear(100, 2)                 # Logits for 2 domains
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        logits = self.classifier(x)  # [batch_size, 2]
        return logits
###########################################################################################



softmax_helper = lambda x: F.softmax(x, 1)

class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    B, S, H, W, C = x.shape
    x = x.view(
        B,
        S // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C,
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows

def window_reverse(windows, window_size, S, H, W):
    B = int(windows.shape[0] / (S * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(
        B,
        S // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, S, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, pos_embed=None):
        B_, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1).contiguous()
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        if pos_embed is not None:
            x = x + pos_embed
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WindowAttention_kv(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, skip, x_up, mask=None):
        B_, N, C = skip.shape
        kv = self.kv(skip)
        q = x_up
        kv = kv.reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q = q.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k, v = kv[0], kv[1]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if tuple(self.input_resolution) == tuple(self.window_size):
            self.shift_size = [0, 0, 0]
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_3tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        S, H, W = self.input_resolution
        assert L == S * H * W, f"input feature has wrong size: L={L}, S*H*W={S*H*W}"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_g = (self.window_size[0] - S % self.window_size[0]) % self.window_size[0]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        _, Sp, Hp, Wp, _ = x.shape
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)
        if min(self.shift_size) > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3)
            )
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()
        x = x.view(B, S * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SwinTransformerBlock_kv(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if tuple(self.input_resolution) == tuple(self.window_size):
            self.shift_size = [0, 0, 0]
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_kv(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix, skip=None, x_up=None):
        assert self.shift_size == [0, 0, 0]
        B, L, C = x.shape
        S, H, W = self.input_resolution
        assert L == S * H * W, "input feature has wrong size"
        shortcut = x
        skip = self.norm1(skip)
        x_up = self.norm1(x_up)
        skip = skip.view(B, S, H, W, C)
        x_up = x_up.view(B, S, H, W, C)
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_g = (self.window_size[0] - S % self.window_size[0]) % self.window_size[0]
        skip = F.pad(skip, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        _, Sp, Hp, Wp, _ = skip.shape
        x_up = F.pad(x_up, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        skip = window_partition(skip, self.window_size)
        skip = skip.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        x_up = window_partition(x_up, self.window_size)
        x_up = x_up.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        attn_windows = self.attn(skip, x_up)
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)
        if min(self.shift_size) > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3)
            )
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()
        x = x.view(B, S * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, tag=None):
        super().__init__()
        self.dim = dim
        if tag == 0:
            self.reduction = nn.Conv3d(dim, dim * 2, kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1])
        elif tag == 1:
            self.reduction = nn.Conv3d(dim, dim * 2, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1])
        else:
            self.reduction = nn.Conv3d(dim, dim * 2, kernel_size=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1])
        self.norm = norm_layer(dim)

    def forward(self, x, S, H, W):
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"
        x = x.view(B, S, H, W, C)
        x = F.gelu(x)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.reduction(x)
        x = x.permute(0, 2, 3, 4, 1).view(B, -1, 2 * C)
        return x

class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, tag=None):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(dim)
        if tag == 0:
            self.up = nn.ConvTranspose3d(dim, dim // 2, [1, 2, 2], [1, 2, 2])
        elif tag == 1:
            self.up = nn.ConvTranspose3d(dim, dim // 2, [2, 2, 2], [2, 2, 2])
        else:  # tag == 2
            self.up = nn.ConvTranspose3d(dim, dim // 2, [2, 2, 2], [2, 2, 2])

    def forward(self, x, S, H, W):
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"
        x = x.view(B, S, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.up(x)
        x = ContiguousGrad.apply(x)
        x = x.permute(0, 2, 3, 4, 1).view(B, -1, C // 2)
        return x

class project(nn.Module):
    def __init__(self, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm2(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        stride1 = [1, patch_size[1] // 2, patch_size[2] // 2]
        stride2 = [1, patch_size[1] // 2, patch_size[2] // 2]
        self.proj1 = project(in_chans, embed_dim // 2, stride1, 1, nn.GELU, nn.LayerNorm, False)
        self.proj2 = project(embed_dim // 2, embed_dim, stride2, 1, nn.GELU, nn.LayerNorm, True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        print(f"PatchEmbed input shape: {x.shape}")
        _, _, S, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if S % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - S % self.patch_size[0]))
        x = self.proj1(x)
        print(f"PatchEmbed after proj1: {x.shape}")
        x = self.proj2(x)
        print(f"PatchEmbed after proj2: {x.shape}")
        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Ws, Wh, Ww)
        print(f"PatchEmbed output shape: {x.shape}")
        return x

class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=True,
        i_layer=None,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = [window_size[0] // 2, window_size[1] // 2, window_size[2] // 2]
        self.depth = depth
        self.i_layer = i_layer
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=[0, 0, 0] if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        if downsample is not None:
            if i_layer == 1:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer, tag=1)
            elif i_layer == 2:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer, tag=2)
            elif i_layer == 0:
                self.downsample = downsample(dim=dim, norm_layer=norm_layer, tag=0)
            else:
                self.downsample = None
        else:
            self.downsample = None

    def forward(self, x, S, H, W):
        print(f"BasicLayer {self.i_layer} input shape: {x.shape}, resolution: ({S}, {H}, {W})")
        attn_mask = None
        for blk in self.blocks:
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, S, H, W)
            print(f"BasicLayer {self.i_layer} after downsample: {x_down.shape}")
            if self.i_layer == 0:
                Ws, Wh, Ww = S, (H + 1) // 2, (W + 1) // 2
            elif self.i_layer == 1:
                Ws, Wh, Ww = S // 2, (H + 1) // 2, (W + 1) // 2
            elif self.i_layer == 2:
                Ws, Wh, Ww = S // 2, (H + 1) // 2, (W + 1) // 2
            else:
                Ws, Wh, Ww = S, H, W
            print(f"BasicLayer {self.i_layer} output resolution: ({Ws}, {Wh}, {Ww})")
            return x, S, H, W, x_down, Ws, Wh, Ww
        else:
            print(f"BasicLayer {self.i_layer} output shape: {x.shape}")
            return x, S, H, W, x, S, H, W

class BasicLayer_up(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        upsample=True,
        i_layer=None,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = [window_size[0] // 2, window_size[1] // 2, window_size[2] // 2]
        self.depth = depth
        self.i_layer = i_layer
        self.blocks = nn.ModuleList()
        self.blocks.append(
            SwinTransformerBlock_kv(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
        )
        for i in range(depth - 1):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i + 1] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
            )
        self.i_layer = i_layer
        if i_layer == 1:
            self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer, tag=1)
        elif i_layer == 0:
            self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer, tag=2)
        else:
            self.Upsample = upsample(dim=2 * dim, norm_layer=norm_layer, tag=0)

    def forward(self, x, skip, S, H, W):
        print(f"BasicLayer_up {self.i_layer} input x shape: {x.shape}, skip shape: {skip.shape}, resolution: ({S}, {H}, {W})")
        x_up = self.Upsample(x, S, H, W)
        print(f"BasicLayer_up {self.i_layer} x_up shape: {x_up.shape}")
        print(f"BasicLayer_up {self.i_layer} before addition: skip shape={skip.shape}, x_up shape={x_up.shape}")
        x = skip + x_up
        print(f"BasicLayer_up {self.i_layer} after addition shape: {x.shape}")
        if self.i_layer == 1:
            S, H, W = S * 2, H * 2, W * 2
        elif self.i_layer == 0:
            S, H, W = S * 2, H * 2, W * 2
        else:
            S, H, W = S, H * 2, W * 2
        print(f"BasicLayer_up {self.i_layer} updated resolution: ({S}, {H}, {W})")
        attn_mask = None
        x = self.blocks[0](x, attn_mask, skip=skip, x_up=x_up)
        print(f"BasicLayer_up {self.i_layer} after first block: {x.shape}")
        for i in range(self.depth - 1):
            x = self.blocks[i + 1](x, attn_mask)
            print(f"BasicLayer_up {self.i_layer} after block {i+1}: {x.shape}")
        return x, S, H, W

class Encoder(nn.Module):
    def __init__(
        self,
        pretrain_img_size=[192, 192, 192],
        patch_size=[1, 4, 4],
        in_chans=1,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[[3, 5, 5], [3, 5, 5], [7, 10, 10], [3, 5, 5]],
        down_stride=[[1, 4, 4], [1, 8, 8], [2, 16, 16], [4, 32, 32]],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
    ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    pretrain_img_size[0] // down_stride[i_layer][0],
                    pretrain_img_size[1] // down_stride[i_layer][1],
                    pretrain_img_size[2] // down_stride[i_layer][2],
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                i_layer=i_layer,
            )
            self.layers.append(layer)
        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

    def forward(self, x):
        print(f"Encoder input shape: {x.shape}")
        x = self.patch_embed(x)
        down = []
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        print(f"Encoder after patch_embed flattened: {x.shape}")
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, S, H, W, x, Ws, Wh, Ww = layer(x, Ws, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)
                out = x_out.view(-1, S, H, W, self.num_features[i]).permute(0, 4, 1, 2, 3).contiguous()
                print(f"Encoder skip {i} shape: {out.shape}")
                down.append(out)
        print(f"Encoder skips: {[s.shape for s in down]}")
        return down

class Decoder(nn.Module):
    def __init__(
        self,
        pretrain_img_size=[192, 192, 192],
        embed_dim=96,
        patch_size=[1, 4, 4],
        depths=[2, 2, 2],
        num_heads=[24, 12, 6],
        window_size=[[3, 5, 5], [7, 10, 10], [3, 5, 5]],
        up_stride=[[4, 32, 32], [2, 16, 16], [1, 8, 8]],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers)[::-1]:
            layer = BasicLayer_up(
                dim=int(embed_dim * 2 ** (len(depths) - i_layer - 1)),
                input_resolution=(
                    pretrain_img_size[0] // up_stride[i_layer][0],
                    pretrain_img_size[1] // up_stride[i_layer][1],
                    pretrain_img_size[2] // up_stride[i_layer][2],
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                upsample=Patch_Expanding,
                i_layer=i_layer,
            )
            self.layers.append(layer)
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]

    def forward(self, x, skips):
        print(f"Decoder input (neck) shape: {x.shape}")
        outs = []
        S, H, W = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        print(f"Decoder after flatten: {x.shape}")
        for index, i in enumerate(skips):
            i = i.flatten(2).transpose(1, 2).contiguous()
            print(f"Decoder skip {index} flattened shape: {i.shape}")
            skips[index] = i
        x = self.pos_drop(x)
        for i in range(self.num_layers)[::-1]:
            layer = self.layers[i]
            print(f"Decoder processing layer {i}")
            x, S, H, W = layer(x, skips[i], S, H, W)
            out = x.view(-1, S, H, W, self.num_features[i])
            print(f"Decoder layer {i} output shape: {out.shape}")
            outs.append(out)
        return outs

class final_patch_expanding(nn.Module):
    def __init__(self, dim, num_class, patch_size):
        super().__init__()
        self.up = nn.ConvTranspose3d(dim, num_class, patch_size, patch_size)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.up(x)
        print(f"final_patch_expanding output shape: {x.shape}")
        return x

class nnFormer3d(nn.Module):
    def __init__(
        self,
        crop_size=[192, 192, 192],
        embedding_dim=96,
        input_channels=1,
        num_classes=1,
        conv_op=nn.Conv3d,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        patch_size=[1, 4, 4],
        window_size=[[3, 5, 5], [3, 5, 5], [7, 10, 10], [3, 5, 5]],
        down_stride=[[1, 4, 4], [1, 8, 8], [2, 16, 16], [4, 32, 32]],
        deep_supervision=False,
        use_domain_adapt=True
    ):
        super().__init__()
        self.use_domain_adapt = use_domain_adapt
        if self.use_domain_adapt:
            self.domain_classifier = DomainClassifier(embed_dims_last=embedding_dim)  
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes = num_classes
        self.conv_op = conv_op
        self.upscale_logits_ops = []
        self.upscale_logits_ops.append(lambda x: x)
        embed_dim = embedding_dim
        depths = depths
        num_heads = num_heads
        patch_size = patch_size
        window_size = window_size
        down_stride = down_stride
        self.model_down = Encoder(
            pretrain_img_size=crop_size,
            window_size=window_size,
            embed_dim=embed_dim,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            in_chans=input_channels,
            down_stride=down_stride,
        )
        self.decoder = Decoder(
            pretrain_img_size=crop_size,
            embed_dim=embed_dim,
            window_size=window_size[::-1][1:],
            patch_size=patch_size,
            num_heads=num_heads[::-1][1:],
            depths=depths[::-1][1:],
            up_stride=down_stride[::-1][1:],
        )
        self.final = []
        self.final.append(final_patch_expanding(embed_dim, num_classes, patch_size=patch_size))
        self.final = nn.ModuleList(self.final)

    # def forward(self, x, return_features=False, use_domain_adapt=False):
    #     print(f"nnFormer3d input shape: {x.shape}")
    #     seg_outputs = []
    #     skips = self.model_down(x)
    #     neck = skips[-1]
    #     print(f"nnFormer3d neck shape: {neck.shape}")
    #     out = self.decoder(neck, skips)
    #     seg_outputs.append(self.final[0](out[-1]))
    #     print(f"nnFormer3d final output shape: {seg_outputs[-1].shape}")
    #     return seg_outputs[-1]


    def forward(self, x, return_features=False, use_domain_adapt=False):
        if x.ndim == 4:
            x = x.unsqueeze(1)  # (B, D, H, W) → (B, 1, D, H, W)

        print(f"nnFormer3d input shape: {x.shape}")
        seg_outputs = []

        # Encoder
        skips = self.model_down(x)
        neck = skips[-1]  # Final encoder features
        print(f"nnFormer3d neck shape: {neck.shape}")

        # Decoder
        out = self.decoder(neck, skips)
        seg_output = self.final[0](out[-1])
        print(f"nnFormer3d final output shape: {seg_output.shape}")

        # Conditional outputs
        if return_features:
            if self.use_domain_adapt and use_domain_adapt and self.domain_classifier is not None:
                reversed_feat = grad_reverse(neck)
                domain_pred = self.domain_classifier(reversed_feat)
                return seg_output, neck, domain_pred
            else:
                return seg_output, neck

        return seg_output

def build_nnformer3d():
    model = nnFormer3d(
        crop_size=[192, 192, 192],
        embedding_dim=96,
        input_channels=1,
        num_classes=1,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        patch_size=[1, 4, 4],
        window_size=[[3, 5, 5], [3, 5, 5], [7, 10, 10], [3, 5, 5]],
        down_stride=[[1, 4, 4], [1, 8, 8], [2, 16, 16], [4, 32, 32]],
        deep_supervision=False,
    )
    return model

def verify_shapes():
    model = build_nnformer3d()
    model.eval()
    
    # Print number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params:,}")
    
    input_tensor = torch.randn(1, 1, 192, 192, 192)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    assert output_tensor.shape == (1,1, 192, 192, 192), "Output shape mismatch!"
    return model, input_tensor, output_tensor

if __name__ == "__main__":
    model, input_tensor, output_tensor = verify_shapes()
