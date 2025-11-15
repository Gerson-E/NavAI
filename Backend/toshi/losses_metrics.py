"""
Loss functions and metrics for kidney segmentation.

This module provides Dice loss, combined BCE+Dice loss, and Dice coefficient
metric for evaluating segmentation performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Compute Dice loss for binary segmentation.
    
    Args:
        pred: Predicted logits tensor of shape (B, 1, H, W)
        target: Target binary mask tensor of shape (B, 1, H, W) with values 0.0 or 1.0
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice loss value (scalar tensor)
    """
    # Apply sigmoid to logits
    pred_probs = torch.sigmoid(pred)
    
    # Flatten tensors
    pred_flat = pred_probs.view(-1)
    target_flat = target.view(-1)
    
    # Compute Dice coefficient
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    # Return Dice loss (1 - Dice coefficient)
    return 1.0 - dice_coeff


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Combined BCE and Dice loss for binary segmentation.
    
    This loss function combines Binary Cross-Entropy (with logits) and Dice loss
    to leverage the benefits of both: BCE provides stable gradients, while Dice
    focuses on the overlap between prediction and target.
    
    Args:
        logits: Predicted logits tensor of shape (B, 1, H, W)
        target: Target binary mask tensor of shape (B, 1, H, W) with values 0.0 or 1.0
        
    Returns:
        Combined loss value (scalar tensor)
    """
    bce = nn.BCEWithLogitsLoss()(logits, target)
    dice = dice_loss(logits, target)
    
    # Weighted combination (equal weights)
    return bce + dice


def dice_coefficient(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> float:
    """
    Compute Dice coefficient metric for binary segmentation.
    
    Both inputs should be binary masks (0 or 1) after thresholding.
    
    Args:
        pred_mask: Predicted binary mask tensor of shape (B, 1, H, W) with values 0 or 1
        target_mask: Target binary mask tensor of shape (B, 1, H, W) with values 0 or 1
        
    Returns:
        Dice coefficient as a float (0.0 to 1.0, higher is better)
    """
    # Flatten tensors
    pred_flat = pred_mask.view(-1).float()
    target_flat = target_mask.view(-1).float()
    
    # Compute Dice coefficient
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    if union == 0:
        return 1.0  # Perfect match if both are empty
    
    dice = (2.0 * intersection) / union
    return dice.item()

