"""
Dataset and dataloader utilities for kidney ultrasound segmentation.

This module provides a PyTorch Dataset class for loading ultrasound images
from the Open Kidney Ultrasound Dataset with CSV-based polygon annotations
from VGG Image Annotator, converting them to binary masks.
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple, Callable, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import cv2


class KidneyUltrasoundDataset(Dataset):
    """
    Dataset for kidney ultrasound images with CSV-based polygon annotations.
    
    Loads grayscale ultrasound images and converts CSV polygon annotations
    (from VGG Image Annotator) to binary masks. Supports the Open Kidney
    Ultrasound Dataset structure.
    
    Args:
        images_dir: Directory containing ultrasound images
        labels_csv: Path to CSV file with VGG Image Annotator annotations
        img_size: Target size for resizing (default: 256)
        transform: Optional transform/augmentation function
        class_name: Class name to extract from annotations (default: 'kidney')
                   Can be 'kidney', 'native_kidney', 'transplant_kidney', etc.
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_csv: str,
        img_size: int = 256,
        transform: Optional[Callable] = None,
        class_name: str = "kidney"
    ):
        self.images_dir = Path(images_dir)
        self.labels_csv = Path(labels_csv)
        self.img_size = img_size
        self.transform = transform
        self.class_name = class_name.lower()
        
        # Load CSV annotations
        print(f"Loading annotations from {self.labels_csv}...")
        self.annotations_df = pd.read_csv(self.labels_csv)
        
        # Get unique image filenames
        if 'filename' in self.annotations_df.columns:
            self.image_files = self.annotations_df['filename'].unique().tolist()
        elif 'file_name' in self.annotations_df.columns:
            self.image_files = self.annotations_df['file_name'].unique().tolist()
        else:
            raise ValueError("CSV must contain 'filename' or 'file_name' column")
        
        # Filter to only images that exist
        self.valid_images = []
        for img_file in self.image_files:
            img_path = self.images_dir / img_file
            if img_path.exists():
                self.valid_images.append(img_file)
            else:
                print(f"Warning: Image not found: {img_file}")
        
        print(f"Found {len(self.valid_images)} valid images with annotations")
    
    def _parse_polygon_coordinates(self, shape_attrs: str) -> Optional[np.ndarray]:
        """
        Parse polygon coordinates from VGG Image Annotator format.
        
        Args:
            shape_attrs: JSON string or dict with shape attributes
            
        Returns:
            Array of polygon points as (N, 2) numpy array, or None if invalid
        """
        try:
            if isinstance(shape_attrs, str):
                attrs = json.loads(shape_attrs)
            else:
                attrs = shape_attrs
            
            if 'all_points_x' in attrs and 'all_points_y' in attrs:
                x_coords = attrs['all_points_x']
                y_coords = attrs['all_points_y']
                if len(x_coords) == len(y_coords) and len(x_coords) >= 3:
                    points = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)
                    return points
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            pass
        
        return None
    
    def _create_mask_from_polygons(
        self,
        polygons: List[np.ndarray],
        img_width: int,
        img_height: int
    ) -> np.ndarray:
        """
        Create a binary mask from a list of polygons.
        
        Args:
            polygons: List of polygon point arrays
            img_width: Image width
            img_height: Image height
            
        Returns:
            Binary mask as numpy array (0 or 255)
        """
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        for polygon in polygons:
            if polygon is not None and len(polygon) >= 3:
                # Use cv2.fillPoly for efficient polygon filling
                cv2.fillPoly(mask, [polygon], 255)
        
        return mask
    
    def _get_annotations_for_image(self, filename: str) -> List[np.ndarray]:
        """
        Get all polygon annotations for a specific image.
        
        Args:
            filename: Image filename
            
        Returns:
            List of polygon point arrays
        """
        # Filter rows for this image
        if 'filename' in self.annotations_df.columns:
            img_rows = self.annotations_df[self.annotations_df['filename'] == filename]
        else:
            img_rows = self.annotations_df[self.annotations_df['file_name'] == filename]
        
        polygons = []
        
        # Check different possible column names for shape attributes
        shape_cols = ['region_shape_attributes', 'shape_attributes']
        region_cols = ['region_attributes', 'region_attr', 'attributes']
        
        for _, row in img_rows.iterrows():
            # Try to find shape attributes column
            shape_attrs = None
            for col in shape_cols:
                if col in row and pd.notna(row[col]):
                    shape_attrs = row[col]
                    break
            
            if shape_attrs is None:
                continue
            
            # Check if this region matches our class (if region_attributes available)
            # For kidney dataset, we'll extract all kidney-related regions
            region_attrs = None
            for col in region_cols:
                if col in row and pd.notna(row[col]):
                    region_attrs = row[col]
                    break
            
            # Parse region attributes if available
            if region_attrs:
                try:
                    if isinstance(region_attrs, str):
                        attrs = json.loads(region_attrs)
                    else:
                        attrs = region_attrs
                    
                    # Check if this region is a kidney (or matches class_name)
                    # VIA format often has class names in region_attributes
                    if isinstance(attrs, dict):
                        # Check if any value contains our class name
                        region_class = None
                        for key, value in attrs.items():
                            if isinstance(value, str) and self.class_name in value.lower():
                                region_class = value
                                break
                        
                        # If we have a specific class filter and it doesn't match, skip
                        if self.class_name != "kidney" and region_class is None:
                            continue
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Parse polygon coordinates
            polygon = self._parse_polygon_coordinates(shape_attrs)
            if polygon is not None:
                polygons.append(polygon)
        
        return polygons
    
    def __len__(self) -> int:
        return len(self.valid_images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single image-mask pair.
        
        Returns:
            Tuple of (image_tensor, mask_tensor), both of shape (1, H, W)
            with dtype float32, normalized to [0, 1]
        """
        img_file = self.valid_images[idx]
        img_path = self.images_dir / img_file
        
        # Load image as grayscale
        image = Image.open(img_path).convert('L')
        original_size = image.size  # (width, height)
        
        # Get polygon annotations for this image
        polygons = self._get_annotations_for_image(img_file)
        
        # Create mask from polygons
        if polygons:
            mask = self._create_mask_from_polygons(
                polygons,
                original_size[0],
                original_size[1]
            )
            mask = Image.fromarray(mask, mode='L')
        else:
            # No annotations found, create empty mask
            mask = Image.new('L', original_size, 0)
        
        # Resize image and mask
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        
        # Convert to numpy arrays
        image_np = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        mask_np = np.array(mask, dtype=np.float32)
        
        # Convert mask: 255 -> 1.0, 0 -> 0.0
        mask_np = (mask_np > 127.5).astype(np.float32)
        
        # Convert to tensors and add channel dimension
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)  # (1, H, W)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # (1, H, W)
        
        # Apply optional transform/augmentation
        if self.transform:
            # Stack for transform (some transforms expect 2-channel or RGB)
            stacked = torch.cat([image_tensor, mask_tensor], dim=0)
            stacked = self.transform(stacked)
            image_tensor = stacked[0:1]
            mask_tensor = stacked[1:2]
        
        return image_tensor, mask_tensor


def get_dataloaders(
    images_dir: str,
    labels_csv: str,
    batch_size: int = 8,
    val_split: float = 0.2,
    img_size: int = 256,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    class_name: str = "kidney"
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for Open Kidney Ultrasound Dataset.
    
    Args:
        images_dir: Directory containing ultrasound images
        labels_csv: Path to CSV file with VGG Image Annotator annotations
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation (0.0 to 1.0)
        img_size: Target size for resizing images
        num_workers: Number of worker processes for data loading
        transform: Optional transform/augmentation function
        class_name: Class name to extract from annotations (default: 'kidney')
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = KidneyUltrasoundDataset(
        images_dir=images_dir,
        labels_csv=labels_csv,
        img_size=img_size,
        transform=transform,
        class_name=class_name
    )
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
