"""
Interactive inference script for U-Net kidney segmentation.

This script prompts the user to upload/select an image and runs inference
using a pre-trained model. No dataset is required - just upload and analyze!
"""

import os
import sys
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

from unet_kidney import UNetKidney
from infer_kidney_unet import (
    preprocess_image,
    postprocess_mask,
    create_overlay,
    detect_kidney_presence
)


def prompt_for_image() -> str:
    """
    Prompt user to enter image path or drag-and-drop.
    
    Returns:
        Path to the image file
    """
    print("\n" + "="*60)
    print("Kidney Ultrasound Segmentation - Interactive Inference")
    print("="*60)
    print("\nPlease provide the path to your ultrasound image:")
    print("  - You can type the full path")
    print("  - Or drag and drop the image file into the terminal")
    print("  - Or type 'quit' to exit")
    print()
    
    while True:
        image_path = input("Image path: ").strip()
        
        # Remove quotes if user dragged and dropped
        image_path = image_path.strip('"').strip("'")
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            sys.exit(0)
        
        if not image_path:
            print("Please enter a valid path.")
            continue
        
        # Expand user home directory
        image_path = os.path.expanduser(image_path)
        
        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            print("Please check the path and try again.")
            continue
        
        # Check if it's an image file
        valid_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.bmp', '.BMP'}
        if Path(image_path).suffix not in valid_extensions:
            print(f"Error: Not a valid image file. Supported: {valid_extensions}")
            continue
        
        return image_path


def prompt_for_model(default_model: str = "kidney_unet.pth") -> str:
    """
    Prompt user for model path or use default.
    
    Args:
        default_model: Default model path
        
    Returns:
        Path to the model file
    """
    print(f"\nModel path (press Enter for default: {default_model}): ", end="")
    model_path = input().strip()
    
    if not model_path:
        model_path = default_model
    
    # Expand user home directory
    model_path = os.path.expanduser(model_path)
    
    if not os.path.exists(model_path):
        print(f"\n⚠️  Warning: Model file not found: {model_path}")
        print("Please make sure you have a trained model.")
        print("You can train one using: python train_kidney_unet.py")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(1)
    
    return model_path


def run_inference(image_path: str, model_path: str, img_size: int = 256, area_threshold: float = 0.01):
    """
    Run inference on a single image.
    
    Args:
        image_path: Path to input image
        model_path: Path to trained model
        img_size: Image size for model input
        area_threshold: Threshold for kidney presence detection
    """
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Using device: {device}")
    
    # Load model
    print(f"📥 Loading model from {model_path}...")
    try:
        model = UNetKidney(in_channels=1, out_channels=1).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Preprocess image
    print(f"\n🖼️  Processing image: {os.path.basename(image_path)}...")
    try:
        image_tensor, original_size = preprocess_image(image_path, img_size)
        image_tensor = image_tensor.to(device)
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return
    
    # Inference
    print("🧠 Running inference...")
    with torch.no_grad():
        logits = model(image_tensor)
    
    # Postprocess mask
    mask = postprocess_mask(logits, original_size)
    
    # Detect kidney presence
    is_present, fraction = detect_kidney_presence(mask, area_threshold)
    
    # Print results
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    print(f"Kidney present: {'✅ YES' if is_present else '❌ NO'}")
    print(f"Mask fraction: {fraction:.4f} ({fraction*100:.2f}% of image)")
    print("="*60)
    
    # Generate output filenames
    base_name = Path(image_path).stem
    output_dir = Path(image_path).parent
    mask_path = output_dir / f"{base_name}_mask.png"
    overlay_path = output_dir / f"{base_name}_overlay.png"
    
    # Save mask
    try:
        mask_image = Image.fromarray(mask)
        mask_image.save(mask_path)
        print(f"\n💾 Saved binary mask to: {mask_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save mask: {e}")
    
    # Create and save overlay
    try:
        overlay = create_overlay(image_path, mask)
        cv2.imwrite(str(overlay_path), overlay)
        print(f"💾 Saved overlay image to: {overlay_path}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save overlay: {e}")
    
    print("\n✅ Inference complete!")


def main():
    """Main interactive loop."""
    print("\n" + "="*60)
    print("Kidney Ultrasound Segmentation - Interactive Mode")
    print("="*60)
    print("\nThis tool analyzes ultrasound images for kidney segmentation.")
    print("No dataset required - just upload your image!\n")
    
    # Check if model exists
    default_model = "kidney_unet.pth"
    if not os.path.exists(default_model):
        print(f"⚠️  Note: Default model '{default_model}' not found.")
        print("   You'll be prompted for the model path.")
        print("   Or train a model first using: python train_kidney_unet.py\n")
    
    # Main loop
    while True:
        try:
            # Prompt for image
            image_path = prompt_for_image()
            
            # Prompt for model
            model_path = prompt_for_model(default_model)
            
            # Run inference
            run_inference(image_path, model_path)
            
            # Ask if user wants to analyze another image
            print("\n" + "-"*60)
            response = input("Analyze another image? (y/n): ").strip().lower()
            if response not in ['y', 'yes']:
                print("\n👋 Thank you for using Kidney Segmentation!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()

