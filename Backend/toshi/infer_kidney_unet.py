"""
Inference script for U-Net kidney segmentation model.

This script loads a trained model and performs inference on a single ultrasound image,
outputting a binary mask, overlay visualization, and kidney presence detection.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from pathlib import Path

from unet_kidney import UNetKidney


def preprocess_image(image_path: str, img_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Preprocess an ultrasound image for inference.
    
    Args:
        image_path: Path to the input image
        img_size: Target size for resizing
        
    Returns:
        Tuple of (preprocessed_tensor, original_size)
    """
    # Load image as grayscale
    image = Image.open(image_path).convert('L')
    original_size = image.size  # (width, height)
    
    # Resize
    image = image.resize((img_size, img_size), Image.BILINEAR)
    
    # Convert to numpy and normalize
    image_np = np.array(image, dtype=np.float32) / 255.0
    
    # Convert to tensor and add batch and channel dimensions
    image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    return image_tensor, original_size


def postprocess_mask(
    logits: torch.Tensor,
    original_size: tuple[int, int],
    threshold: float = 0.5
) -> np.ndarray:
    """
    Postprocess model output to get binary mask.
    
    Args:
        logits: Model output logits of shape (1, 1, H, W)
        original_size: Original image size (width, height)
        threshold: Threshold for binarization (default: 0.5)
        
    Returns:
        Binary mask as numpy array of shape (H, W) with values 0 or 255
    """
    # Apply sigmoid
    probs = torch.sigmoid(logits)
    
    # Threshold
    mask = (probs > threshold).float()
    
    # Convert to numpy
    mask_np = mask.squeeze().cpu().numpy()  # (H, W)
    
    # Resize to original size
    mask_np = cv2.resize(mask_np, original_size, interpolation=cv2.INTER_NEAREST)
    
    # Convert to 0/255
    mask_np = (mask_np * 255).astype(np.uint8)
    
    return mask_np


def create_overlay(original_image_path: str, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Create an overlay image with kidney region highlighted in red.
    
    Args:
        original_image_path: Path to original image
        mask: Binary mask (0 or 255)
        alpha: Transparency factor for overlay (default: 0.5)
        
    Returns:
        Overlay image as BGR numpy array
    """
    # Load original image in BGR format
    original = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert to BGR for color overlay
    original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    # Create red overlay where mask is 255
    mask_bool = mask > 127
    overlay = original_bgr.copy()
    overlay[mask_bool] = [0, 0, 255]  # Red in BGR
    
    # Blend original and overlay
    result = cv2.addWeighted(original_bgr, 1.0 - alpha, overlay, alpha, 0)
    
    return result


def detect_kidney_presence(mask: np.ndarray, area_threshold: float = 0.01) -> tuple[bool, float]:
    """
    Determine if a kidney is present based on mask area.
    
    Args:
        mask: Binary mask (0 or 255)
        area_threshold: Minimum fraction of pixels that must be kidney (default: 0.01 = 1%)
        
    Returns:
        Tuple of (is_present, fraction)
    """
    total_pixels = mask.size
    kidney_pixels = np.sum(mask > 127)
    fraction = kidney_pixels / total_pixels if total_pixels > 0 else 0.0
    
    is_present = fraction >= area_threshold
    
    return is_present, fraction


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single ultrasound image")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input ultrasound image")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--out_mask", type=str, default="kidney_mask.png",
                        help="Path to save binary mask (default: kidney_mask.png)")
    parser.add_argument("--out_overlay", type=str, default="kidney_overlay.png",
                        help="Path to save overlay image (default: kidney_overlay.png)")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image size for model input (default: 256)")
    parser.add_argument("--area_threshold", type=float, default=0.01,
                        help="Minimum fraction of pixels for kidney presence (default: 0.01)")
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = UNetKidney(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print("Model loaded successfully.")
    
    # Preprocess image
    print(f"Processing image: {args.image}...")
    image_tensor, original_size = preprocess_image(args.image, args.img_size)
    image_tensor = image_tensor.to(device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        logits = model(image_tensor)
    
    # Postprocess mask
    mask = postprocess_mask(logits, original_size)
    
    # Detect kidney presence
    is_present, fraction = detect_kidney_presence(mask, args.area_threshold)
    
    # Print results
    print("\n" + "="*50)
    print(f"Kidney present: {is_present}")
    print(f"Mask fraction: {fraction:.6f}")
    print("="*50)
    
    # Save mask
    mask_image = Image.fromarray(mask)
    mask_image.save(args.out_mask)
    print(f"\nSaved binary mask to: {args.out_mask}")
    
    # Create and save overlay
    overlay = create_overlay(args.image, mask)
    cv2.imwrite(args.out_overlay, overlay)
    print(f"Saved overlay image to: {args.out_overlay}")


if __name__ == "__main__":
    main()

