"""
Training script for U-Net kidney segmentation model.

Designed for the Open Kidney Ultrasound Dataset with CSV-based annotations.

Example usage:
    python train_kidney_unet.py \\
      --images_dir data/ultrasound/images \\
      --labels_csv labels/annotations.csv \\
      --epochs 30 \\
      --out kidney_unet.pth
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from unet_kidney import UNetKidney
from dataset_kidney import get_dataloaders
from losses_metrics import bce_dice_loss, dice_coefficient


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for images, masks in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(images)
        loss = bce_dice_loss(logits, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> tuple[float, float]:
    """Validate and return average loss and Dice score."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            
            # Compute Dice score
            pred_probs = torch.sigmoid(logits)
            pred_mask = (pred_probs > 0.5).float()
            dice = dice_coefficient(pred_mask, masks)
            
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_dice = total_dice / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_dice


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for kidney segmentation")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing ultrasound images")
    parser.add_argument("--labels_csv", type=str, required=True,
                        help="Path to CSV file with VGG Image Annotator annotations")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image size for resizing (default: 256)")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers (default: 4)")
    parser.add_argument("--class_name", type=str, default="kidney",
                        help="Class name to extract from annotations (default: 'kidney')")
    parser.add_argument("--out", type=str, default="kidney_unet.pth",
                        help="Path to save model checkpoint (default: kidney_unet.pth)")
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = get_dataloaders(
        images_dir=args.images_dir,
        labels_csv=args.labels_csv,
        batch_size=args.batch_size,
        val_split=args.val_split,
        img_size=args.img_size,
        num_workers=args.num_workers,
        class_name=args.class_name
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize model
    model = UNetKidney(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_val_dice = 0.0
    
    print("\nStarting training...")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val Dice':<12} {'Best':<8}")
    print("-" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, device)
        
        # Check if best model
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
        
        # Print metrics
        best_marker = "✓" if is_best else ""
        print(f"{epoch:<8} {train_loss:<12.6f} {val_loss:<12.6f} {val_dice:<12.6f} {best_marker:<8}")
        
        # Save model (only when validation Dice improves)
        if is_best:
            torch.save(model.state_dict(), args.out)
            print(f"  → Saved model to {args.out} (Dice: {val_dice:.6f})")
    
    print("\nTraining completed!")
    print(f"Best validation Dice: {best_val_dice:.6f}")
    print(f"Final model saved to: {args.out}")


if __name__ == "__main__":
    main()

