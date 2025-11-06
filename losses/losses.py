import torch
import monai
import torch.nn as nn
from typing import Dict
from monai import losses

import torch
import torch.nn as nn
from monai.losses import DiceCELoss, FocalLoss


class DiceLoss(nn.Module): #This is actually not just dice it also  has integartion of DiceCE and Focal Loss
    """
    Combines Dice+CE, Focal, and Surface losses for 3D QCT segmentation.

    total_loss = w_base * DiceCELoss
               + alpha * FocalLoss
               + beta  * SurfaceLoss
    where w_base = 1 - alpha - beta
    """
    def __init__(
        self,
        alpha: float = 0.5,         # weight for FocalLoss
        beta: float  = 0.1,         # weight for SurfaceLoss
        gamma: float = 2.0,         # focusing exponent in FocalLoss
        lambda_dice: float = 1.0,   # weight of Dice in DiceCELoss
        lambda_ce: float   = 1.0,   # weight of CE  in DiceCELoss
    ):
        super().__init__()
        assert alpha + beta < 1.0, "alpha + beta must be < 1"
        self.alpha = alpha
        self.beta  = beta
        self.w_base = 1.0 - alpha

        # Dice + CE term
        self.dice_ce = losses.DiceCELoss(to_onehot_y=False, sigmoid=True)
        # Focal term
        self.focal = losses.FocalLoss(to_onehot_y=False)


    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (B,1,D,H,W) logits
        target: (B,1,D,H,W) or (B,D,H,W) binary {0,1}
        """
        loss_base    = self.dice_ce(pred, target)
        loss_focal   = self.focal(pred, target)

        return (
            self.w_base    * loss_base
          + self.alpha     * loss_focal
        )


###########################################################################
def build_loss_fn(loss_type: str, loss_args: Dict = None):
    if loss_type == "dice":
        return DiceLoss() #here even dice i called underlying imeplementation is diceCe
    else:
        raise ValueError("Defined Loss Not Avilable")
