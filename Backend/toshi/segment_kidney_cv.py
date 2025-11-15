"""
Classical computer vision-based kidney segmentation for ultrasound images.

This script uses OpenCV edge detection, contour analysis, and morphological
operations to segment kidney regions without requiring machine learning models.

This is a prototype implementation that will be replaced with a CNN/U-Net later.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_gray(path: str) -> np.ndarray:
    """
    Load an image as grayscale.
    
    Args:
        path: Path to the image file
        
    Returns:
        Grayscale image as uint8 numpy array
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the image cannot be loaded
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from: {path}. Check file format.")
    
    return img


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Enhance contrast of grayscale image using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        gray: Input grayscale image
        
    Returns:
        Contrast-enhanced grayscale image
    """
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Apply CLAHE
    enhanced = clahe.apply(gray)
    
    return enhanced


def find_kidney_mask(
    gray: np.ndarray,
    canny_low: int = 30,
    canny_high: int = 80,
    min_frac: float = 0.01,
    max_frac: float = 0.65,
) -> np.ndarray:
    """
    Find kidney mask using multiple approaches: thresholding, edge detection, and contour analysis.
    
    Args:
        gray: Input grayscale image
        canny_low: Lower threshold for Canny edge detection
        canny_high: Upper threshold for Canny edge detection
        min_frac: Minimum fraction of image area for valid contour (default: 0.01 = 1%)
        max_frac: Maximum fraction of image area for valid contour (default: 0.4 = 40%)
        
    Returns:
        Binary mask (uint8) with 255 for kidney region, 0 for background
    """
    h, w = gray.shape
    total_pixels = h * w
    min_area = int(total_pixels * min_frac)
    max_area = int(total_pixels * max_frac)
    
    
    # Create output mask (all zeros initially)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Step 1: Enhanced preprocessing with multi-scale approach
    # Apply non-local means denoising for ultrasound (better than bilateral for speckle)
    try:
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    except:
        # Fallback to bilateral if fastNlMeansDenoising not available
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply bilateral filter to further reduce noise while preserving edges
    filtered = cv2.bilateralFilter(denoised, 9, 75, 75)
    
    # Multi-scale Gaussian blur for different feature scales
    blurred_fine = cv2.GaussianBlur(filtered, (3, 3), 0)
    blurred = cv2.GaussianBlur(filtered, (5, 5), 0)
    blurred_coarse = cv2.GaussianBlur(filtered, (7, 7), 0)
    
    # Step 2: Multi-scale approach - try different methods
    # IMPORTANT: Kidneys typically have a bright capsule/cortex and dark interior (renal pelvis)
    # We need to detect the ENTIRE kidney structure, not just the dark center
    candidates = []
    
    # EDGE-BASED METHODS (Primary for detecting kidney boundary/capsule)
    # Method 1: Canny edge detection with multiple sensitivity levels
    edges_tight = cv2.Canny(blurred, canny_low, canny_high)
    edges_loose = cv2.Canny(blurred, canny_low // 2, canny_high // 2)
    edges_very_loose = cv2.Canny(blurred, max(10, canny_low // 3), max(20, canny_high // 3))
    edges_ultra_loose = cv2.Canny(blurred, 5, 15)  # Very sensitive
    
    # Close edges to form complete boundaries - VERY aggressive
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_xlarge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_xxlarge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    kernel_xxxlarge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    
    edges_closed_tight = cv2.morphologyEx(edges_tight, cv2.MORPH_CLOSE, kernel_large, iterations=6)
    edges_closed_loose = cv2.morphologyEx(edges_loose, cv2.MORPH_CLOSE, kernel_xlarge, iterations=8)
    edges_closed_very_loose = cv2.morphologyEx(edges_very_loose, cv2.MORPH_CLOSE, kernel_xxlarge, iterations=12)
    edges_closed_ultra_loose = cv2.morphologyEx(edges_ultra_loose, cv2.MORPH_CLOSE, kernel_xxxlarge, iterations=15)
    
    # Fill enclosed regions
    edges_filled_tight = edges_closed_tight.copy()
    edges_filled_loose = edges_closed_loose.copy()
    edges_filled_very_loose = edges_closed_very_loose.copy()
    edges_filled_ultra_loose = edges_closed_ultra_loose.copy()
    
    contours_temp, _ = cv2.findContours(edges_closed_tight, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edges_filled_tight, contours_temp, -1, 255, thickness=cv2.FILLED)
    contours_temp, _ = cv2.findContours(edges_closed_loose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edges_filled_loose, contours_temp, -1, 255, thickness=cv2.FILLED)
    contours_temp, _ = cv2.findContours(edges_closed_very_loose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edges_filled_very_loose, contours_temp, -1, 255, thickness=cv2.FILLED)
    contours_temp, _ = cv2.findContours(edges_closed_ultra_loose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edges_filled_ultra_loose, contours_temp, -1, 255, thickness=cv2.FILLED)
    
    candidates.append(("canny_filled_tight", edges_filled_tight))
    candidates.append(("canny_filled_loose", edges_filled_loose))
    candidates.append(("canny_filled_very_loose", edges_filled_very_loose))
    candidates.append(("canny_filled_ultra_loose", edges_filled_ultra_loose))
    
    # Method 2: Gradient-based boundary detection
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    
    # Threshold gradient to get strong boundaries
    _, gradient_strong = cv2.threshold(gradient_magnitude, np.percentile(gradient_magnitude, 75), 255, cv2.THRESH_BINARY)
    gradient_closed = cv2.morphologyEx(gradient_strong, cv2.MORPH_CLOSE, kernel_large, iterations=5)
    
    # Fill enclosed regions
    gradient_filled = gradient_closed.copy()
    contours_temp, _ = cv2.findContours(gradient_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(gradient_filled, contours_temp, -1, 255, thickness=cv2.FILLED)
    candidates.append(("gradient_filled", gradient_filled))
    
    # Method 3: Combined edge + gradient approach
    combined_edges = cv2.bitwise_or(edges_tight, gradient_strong)
    combined_closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel_large, iterations=6)
    combined_filled = combined_closed.copy()
    contours_temp, _ = cv2.findContours(combined_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(combined_filled, contours_temp, -1, 255, thickness=cv2.FILLED)
    candidates.append(("combined_edges_filled", combined_filled))
    
    # REGION-BASED METHODS (Secondary, for comparison)
    # Method 4: Adaptive thresholding
    adaptive_thresh_dark = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 3
    )
    adaptive_closed = cv2.morphologyEx(adaptive_thresh_dark, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
    candidates.append(("adaptive_closed", adaptive_closed))
    
    # Method 5: Otsu's thresholding (for comparison)
    _, otsu_dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates.append(("otsu_dark", otsu_dark))
    
    # Method 6: Morphological gradient (finds boundaries)
    morph_gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel_medium)
    _, morph_thresh = cv2.threshold(morph_gradient, np.percentile(morph_gradient, 70), 255, cv2.THRESH_BINARY)
    morph_closed = cv2.morphologyEx(morph_thresh, cv2.MORPH_CLOSE, kernel_large, iterations=6)
    morph_filled = morph_closed.copy()
    contours_temp, _ = cv2.findContours(morph_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(morph_filled, contours_temp, -1, 255, thickness=cv2.FILLED)
    candidates.append(("morph_gradient_filled", morph_filled))
    
    # Method 7: Multi-threshold approach - find regions enclosed by bright boundaries
    # Find bright regions (potential kidney capsule)
    _, bright_regions = cv2.threshold(blurred, np.percentile(blurred, 60), 255, cv2.THRESH_BINARY)
    bright_edges = cv2.morphologyEx(bright_regions, cv2.MORPH_GRADIENT, kernel_medium)
    bright_closed = cv2.morphologyEx(bright_edges, cv2.MORPH_CLOSE, kernel_large, iterations=7)
    bright_filled = bright_closed.copy()
    contours_temp, _ = cv2.findContours(bright_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bright_filled, contours_temp, -1, 255, thickness=cv2.FILLED)
    candidates.append(("bright_boundary_filled", bright_filled))
    
    # Method 8: Combined multi-method approach
    # Combine multiple edge detection results
    combined_multi = cv2.bitwise_or(edges_closed_tight, edges_closed_loose)
    combined_multi = cv2.bitwise_or(combined_multi, gradient_strong)
    combined_multi = cv2.bitwise_or(combined_multi, morph_thresh)
    combined_multi_closed = cv2.morphologyEx(combined_multi, cv2.MORPH_CLOSE, kernel_xlarge, iterations=8)
    combined_multi_filled = combined_multi_closed.copy()
    contours_temp, _ = cv2.findContours(combined_multi_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(combined_multi_filled, contours_temp, -1, 255, thickness=cv2.FILLED)
    candidates.append(("combined_multi_filled", combined_multi_filled))
    
    # Method 9: Super aggressive - combine ALL edge methods
    combined_all = cv2.bitwise_or(edges_closed_tight, edges_closed_loose)
    combined_all = cv2.bitwise_or(combined_all, edges_closed_very_loose)
    combined_all = cv2.bitwise_or(combined_all, gradient_strong)
    combined_all = cv2.bitwise_or(combined_all, morph_thresh)
    combined_all = cv2.bitwise_or(combined_all, bright_edges)
    
    # Extra aggressive closing
    kernel_xxlarge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    combined_all_closed = cv2.morphologyEx(combined_all, cv2.MORPH_CLOSE, kernel_xxlarge, iterations=10)
    combined_all_filled = combined_all_closed.copy()
    contours_temp, _ = cv2.findContours(combined_all_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(combined_all_filled, contours_temp, -1, 255, thickness=cv2.FILLED)
    candidates.append(("combined_all_aggressive", combined_all_filled))
    
    # Method 10: Convex hull approach - find largest contour and compute its convex hull
    for candidate_name, candidate_img in [("edges_very_loose", edges_closed_very_loose), 
                                          ("combined_multi", combined_multi_closed)]:
        contours_hull, _ = cv2.findContours(candidate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_hull:
            # Get largest contour
            largest_contour = max(contours_hull, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > min_area:
                # Compute convex hull
                hull = cv2.convexHull(largest_contour)
                hull_mask = np.zeros_like(candidate_img)
                cv2.fillPoly(hull_mask, [hull], 255)
                candidates.append((f"{candidate_name}_convex_hull", hull_mask))
    
    # Step 3: Process each candidate and find best kidney region
    best_contour = None
    best_score = -1
    best_method = None
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    
    for method_name, candidate_img in candidates:
        # Morphological operations to clean up
        cleaned = cv2.morphologyEx(candidate_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
        
        # Evaluate each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            area_frac = area / total_pixels
            
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Get bounding box
            x, y, w_box, h_box = cv2.boundingRect(contour)
            if w_box == 0 or h_box == 0:
                continue
            
            # Calculate features for scoring
            aspect_ratio = max(w_box, h_box) / min(w_box, h_box)
            extent = area / (w_box * h_box)
            
            # Calculate solidity (area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Calculate circularity (4π*area / perimeter^2)
            perimeter = cv2.arcLength(contour, True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Score the contour (kidney should be reasonably compact, not too circular, not too elongated)
            score = 0
            
            # Prefer moderate to high aspect ratios (kidney is often elongated, can be diagonal)
            # Kidneys can be quite elongated, especially in certain views
            if 1.5 <= aspect_ratio <= 4.5:  # More lenient for elongated kidneys
                score += 3
            elif 1.2 <= aspect_ratio <= 5.0:
                score += 2
            elif 1.0 <= aspect_ratio <= 6.0:
                score += 1
            
            # Prefer good extent (not too irregular)
            # Kidneys can be somewhat irregular, so be more lenient
            if extent > 0.5:
                score += 3
            elif extent > 0.4:
                score += 2
            elif extent > 0.3:
                score += 1
            elif extent > 0.25:  # Allow more irregular shapes
                score += 0.5
            
            # Prefer good solidity (not too concave)
            # Kidneys can have some concavity (renal hilum), so be lenient
            if solidity > 0.75:
                score += 3
            elif solidity > 0.65:
                score += 2
            elif solidity > 0.55:
                score += 1
            elif solidity > 0.45:  # Allow some concavity
                score += 0.5
            
            # Prefer moderate circularity (kidney is not perfectly round)
            if 0.3 <= circularity <= 0.7:
                score += 1
            
            # Prefer contours in central region (kidney usually in center of ultrasound)
            center_x, center_y = x + w_box/2, y + h_box/2
            img_center_x, img_center_y = w/2, h/2
            dist_from_center = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            max_dist = np.sqrt(w**2 + h**2) / 2
            if dist_from_center < max_dist * 0.4:
                score += 1
            
            # Normalize by area (prefer larger regions for complete kidney with capsule)
            # Edge-based methods tend to find larger, more complete structures
            area_frac = area / total_pixels
            
            # Bonus for edge-based methods (they find complete kidney boundaries)
            is_edge_method = any(keyword in method_name for keyword in 
                                ['canny', 'gradient', 'morph_gradient', 'combined', 'convex_hull', 'bright_boundary'])
            is_region_method = any(keyword in method_name for keyword in 
                                  ['otsu', 'adaptive', 'percentile'])
            
            # Edge methods should be strongly preferred for larger regions
            if 0.15 <= area_frac <= 0.45:  # 15-45% of image (optimal complete kidney)
                score += 8 if is_edge_method else 5
            elif 0.12 <= area_frac <= 0.48:  # 12-48% of image
                score += 7 if is_edge_method else 4
            elif 0.10 <= area_frac <= 0.50:  # 10-50% of image
                score += 6 if is_edge_method else 3
            elif 0.08 <= area_frac <= 0.50:
                score += 4 if is_edge_method else 2
            elif 0.05 <= area_frac <= 0.50:
                score += 2 if is_edge_method else 1
            elif 0.03 <= area_frac <= 0.50:
                score += 1
            elif area_frac < 0.03:  # Too small - heavily penalize
                score -= 5
            
            # Penalize region-based methods for being too small (they tend to find only dark interior)
            if is_region_method and area_frac < 0.15:  # Less than 15% is too small for complete kidney
                score -= 5  # Heavy penalty
            
            # Penalize very large regions (likely ultrasound cone, not kidney)
            if area_frac > 0.65:  # Too large
                score -= 5
            elif area_frac > 0.60:
                score -= 3
            
            
            # Additional: Check texture homogeneity (kidney should have relatively uniform texture)
            # Extract region from original image
            roi_mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(roi_mask, [contour], 255)
            roi_pixels = gray[roi_mask > 127]
            
            if len(roi_pixels) > 100:  # Only if region is large enough
                roi_std = np.std(roi_pixels)
                roi_mean = np.mean(roi_pixels)
                cv_value = roi_std / roi_mean if roi_mean > 0 else 0
                
                # Prefer regions with moderate coefficient of variation (not too uniform, not too noisy)
                if 0.15 <= cv_value <= 0.5:
                    score += 1
                
                # Additional: Check local contrast (kidney should have distinct boundaries)
                # Get pixels just outside the contour
                dilated_roi = cv2.dilate(roi_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
                boundary_region = cv2.subtract(dilated_roi, roi_mask)
                boundary_pixels = gray[boundary_region > 127]
                
                if len(boundary_pixels) > 50:
                    boundary_mean = np.mean(boundary_pixels)
                    contrast = abs(roi_mean - boundary_mean) / 255.0
                    
                    # Prefer regions with good contrast to surroundings
                    # For dark kidneys (hypoechoic), contrast should be high
                    # For bright kidneys (hyperechoic), contrast can vary
                    if contrast > 0.15:  # At least 15% intensity difference
                        score += 2
                        if contrast > 0.25:  # Strong contrast (very important for dark kidneys)
                            score += 2
                    elif contrast > 0.08:  # Moderate contrast
                        score += 1
                    
                    # Bonus for dark regions with bright surroundings (hypoechoic kidney)
                    if roi_mean < boundary_mean and contrast > 0.2:
                        score += 1  # Dark structure with bright wall is kidney-like
                
                # Additional: Check for kidney-like intensity distribution
                # Kidneys often have a bimodal or skewed distribution
                hist, _ = np.histogram(roi_pixels, bins=32)
                hist = hist / hist.sum() if hist.sum() > 0 else hist
                
                # Check for peak concentration (kidney tissue tends to cluster)
                peak_concentration = np.max(hist)
                if peak_concentration > 0.20:  # At least 20% of pixels in one bin
                    score += 2
                elif peak_concentration > 0.15:
                    score += 1
                
                # Check for kidney-like intensity pattern (mixed: bright capsule + dark interior)
                # Calculate intensity distribution characteristics
                roi_min = np.min(roi_pixels)
                roi_max = np.max(roi_pixels)
                roi_range = roi_max - roi_min
                
                # Kidneys typically have good internal contrast (bright capsule + dark interior)
                if roi_range > 60:  # Excellent internal contrast
                    score += 4
                elif roi_range > 50:  # Good internal contrast
                    score += 3
                elif roi_range > 40:
                    score += 2
                elif roi_range > 30:
                    score += 1
                
                # Check for mixed intensity (not all dark, not all bright)
                dark_ratio = np.sum(roi_pixels < np.percentile(gray, 40)) / len(roi_pixels)
                bright_ratio = np.sum(roi_pixels > np.percentile(gray, 60)) / len(roi_pixels)
                
                # Prefer regions with both dark and bright areas (typical kidney structure)
                if 0.25 <= dark_ratio <= 0.70 and 0.25 <= bright_ratio <= 0.70:
                    score += 4  # Excellent mix of dark and bright
                elif 0.20 <= dark_ratio <= 0.75 and 0.20 <= bright_ratio <= 0.75:
                    score += 3
                elif dark_ratio > 0.3 and bright_ratio > 0.3:
                    score += 2
                elif dark_ratio > 0.2 and bright_ratio > 0.2:
                    score += 1
            
            # Update best candidate
            if score > best_score:
                best_score = score
                best_contour = contour
                best_method = method_name
    
    # If we found a good candidate, use it
    # Lower threshold to allow more detections
    if best_contour is not None and best_score >= 3:
        cv2.fillPoly(mask, [best_contour], 255)
        area = cv2.contourArea(best_contour)
        print(f"  Selected contour ({best_method}, score: {best_score}) with area: {area:.0f} pixels")
        
        # Advanced refinement: try to improve the mask using region growing
        mask = refine_mask_advanced(mask, gray, blurred)
        
        return mask
    
    # Fallback to simpler approach
    print("  No high-scoring contours found, trying simpler approach...")
    return find_kidney_mask_simple(gray, min_frac, max_frac)


def refine_mask_advanced(mask: np.ndarray, original_gray: np.ndarray, blurred: np.ndarray) -> np.ndarray:
    """
    Advanced mask refinement using conservative region growing and intelligent boundary adjustment.
    
    Args:
        mask: Initial binary mask
        original_gray: Original grayscale image
        blurred: Preprocessed blurred image
        
    Returns:
        Refined binary mask
    """
    if np.sum(mask > 127) == 0:
        return mask
    
    # Start with basic refinement
    refined = refine_mask(mask, original_gray)
    
    # Get initial area to prevent excessive growth
    initial_area = np.sum(refined > 127)
    max_growth_factor = 1.3  # Allow up to 30% growth
    
    # Advanced: Conservative region growing from refined mask
    # Get statistics of the refined region
    region_pixels = original_gray[refined > 127]
    if len(region_pixels) > 0:
        mean_int = np.mean(region_pixels)
        std_int = np.std(region_pixels)
        
        # Create a narrow band around the current mask (more conservative)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(refined, kernel, iterations=1)  # Only 1 iteration
        boundary_band = cv2.subtract(dilated, refined)
        
        # In the boundary band, include pixels that are similar to the region
        # Use conservative threshold (1.5 std instead of 2.0)
        similar_mask = np.abs(original_gray - mean_int) < (std_int * 1.5)
        similar_mask = similar_mask.astype(np.uint8) * 255
        
        # Also check gradient - don't grow across strong edges
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_thresh = np.percentile(gradient_mag, 70)  # Top 30% gradients
        weak_edges = (gradient_mag < gradient_thresh).astype(np.uint8) * 255
        
        # Combine: similar intensity AND weak edges
        candidate_pixels = cv2.bitwise_and(similar_mask, weak_edges)
        candidate_pixels = cv2.bitwise_and(candidate_pixels, boundary_band)
        
        # Combine refined mask with candidate pixels
        grown_mask = cv2.bitwise_or(refined, candidate_pixels)
        
        # Check if growth is reasonable
        new_area = np.sum(grown_mask > 127)
        if new_area > initial_area * max_growth_factor:
            # Too much growth, use original refined mask
            grown_mask = refined
        
        # Clean up: remove small isolated regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(grown_mask, connectivity=8)
        if num_labels > 1:
            # Keep only the largest component
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            final_mask = np.zeros_like(grown_mask)
            final_mask[labels == largest_label] = 255
            
            # Final check: ensure we didn't grow too much
            final_area = np.sum(final_mask > 127)
            if final_area > initial_area * max_growth_factor:
                return refined  # Return original if too much growth
            
            return final_mask
        
        return grown_mask
    
    return refined


def refine_mask(mask: np.ndarray, original_gray: np.ndarray) -> np.ndarray:
    """
    Refine the mask using advanced processing steps including boundary refinement.
    
    Args:
        mask: Initial binary mask
        original_gray: Original grayscale image
        
    Returns:
        Refined binary mask
    """
    if np.sum(mask > 127) == 0:
        return mask
    
    # Step 1: Remove small holes inside the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Step 2: Remove small isolated regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels > 1:
        # Find the largest component (should be the kidney)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # Create mask with only the largest component
        refined_mask = np.zeros_like(mask)
        refined_mask[labels == largest_label] = 255
        mask = refined_mask
    
    # Step 3: Boundary refinement using gradient information
    # Find edges in the original image
    edges = cv2.Canny(original_gray, 50, 150)
    
    # Dilate edges slightly
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, edge_kernel, iterations=1)
    
    # Use edges to refine mask boundary
    # Keep mask pixels that are near strong edges (likely boundaries)
    mask_boundary = cv2.Canny(mask, 50, 150)
    
    # Combine: keep mask where it overlaps with image edges or where mask boundary is strong
    combined_edges = cv2.bitwise_or(edges_dilated, mask_boundary)
    
    # Step 4: Active contour-like refinement
    # Get contour of current mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # Find the main contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Create a refined mask by considering local image statistics
        refined_mask = np.zeros_like(mask)
        
        # Get region statistics
        mask_region = original_gray[mask > 127]
        if len(mask_region) > 0:
            mean_intensity = np.mean(mask_region)
            std_intensity = np.std(mask_region)
            
            # Create a region around the contour
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [main_contour], -1, 255, -1)
            
            # Dilate to get a region around the boundary
            dilated_contour = cv2.dilate(contour_mask, kernel, iterations=3)
            boundary_region = cv2.subtract(dilated_contour, contour_mask)
            
            # In boundary region, include pixels with similar intensity
            boundary_pixels = original_gray[boundary_region > 127]
            if len(boundary_pixels) > 0:
                similar_pixels = np.abs(original_gray - mean_intensity) < (std_intensity * 1.5)
                similar_pixels = similar_pixels.astype(np.uint8) * 255
                
                # Combine original mask with similar pixels in boundary region
                refined_mask = cv2.bitwise_or(contour_mask, 
                                             cv2.bitwise_and(similar_pixels, boundary_region))
            else:
                refined_mask = contour_mask
            
            mask = refined_mask
    
    # Step 5: Final smoothing
    # Smooth the boundary
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Final cleanup: remove tiny holes and protrusions
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask


def find_kidney_mask_simple(gray: np.ndarray, min_frac: float, max_frac: float) -> np.ndarray:
    """
    Fallback simple approach using multiple thresholding strategies.
    
    Args:
        gray: Input grayscale image
        min_frac: Minimum fraction of image area
        max_frac: Maximum fraction of image area
        
    Returns:
        Binary mask (uint8) with 255 for kidney region, 0 for background
    """
    h, w = gray.shape
    total_pixels = h * w
    min_area = int(total_pixels * min_frac)
    max_area = int(total_pixels * max_frac)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Try multiple thresholding approaches with better scoring
    approaches = []
    
    # Approach 1: Otsu's threshold (inverted - looking for dark regions)
    _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    approaches.append(("Otsu (dark)", thresh1))
    
    # Approach 2: Otsu's threshold (normal - looking for bright regions)
    _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    approaches.append(("Otsu (bright)", thresh2))
    
    # Approach 3: Adaptive threshold (dark regions) - multiple block sizes
    for block_size in [11, 15, 21]:
        adaptive1 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 3
        )
        approaches.append((f"Adaptive dark ({block_size})", adaptive1))
    
    # Approach 4: Adaptive threshold (bright regions)
    for block_size in [11, 15, 21]:
        adaptive2 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 3
        )
        approaches.append((f"Adaptive bright ({block_size})", adaptive2))
    
    # Approach 5: Percentile-based thresholds (multiple ranges)
    p10 = np.percentile(blurred, 10)
    p25 = np.percentile(blurred, 25)
    p50 = np.percentile(blurred, 50)
    p75 = np.percentile(blurred, 75)
    p90 = np.percentile(blurred, 90)
    
    approaches.append(("Percentile low", cv2.inRange(blurred, p10, p50)))
    approaches.append(("Percentile mid", cv2.inRange(blurred, p25, p75)))
    approaches.append(("Percentile high", cv2.inRange(blurred, p50, p90)))
    
    # Score and select best contour
    best_contour = None
    best_score = -1
    best_method = None
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    for name, thresh_img in approaches:
        # Morphological cleanup
        cleaned = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
        
        # Evaluate each contour with scoring
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < min_area:
                continue
            
            # Get bounding box
            x, y, w_box, h_box = cv2.boundingRect(contour)
            if w_box == 0 or h_box == 0:
                continue
            
            # Calculate features
            aspect_ratio = max(w_box, h_box) / min(w_box, h_box)
            extent = area / (w_box * h_box)
            
            # Calculate solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Score the contour
            score = 0
            
            # Prefer reasonable aspect ratios
            if 1.2 <= aspect_ratio <= 3.5:
                score += 2
            elif 1.0 <= aspect_ratio <= 4.0:
                score += 1
            
            # Prefer good extent
            if extent > 0.4:
                score += 2
            elif extent > 0.3:
                score += 1
            
            # Prefer good solidity
            if solidity > 0.7:
                score += 2
            elif solidity > 0.6:
                score += 1
            
            # Prefer medium-sized regions (penalize very large)
            area_frac = area / total_pixels
            if 0.03 <= area_frac <= 0.20:  # Optimal range
                score += 3
            elif 0.02 <= area_frac <= 0.25:
                score += 2
            elif 0.01 <= area_frac <= 0.30:
                score += 1
            elif area_frac > 0.35:  # Too large, penalize
                score -= 2
            
            # Update best
            if score > best_score:
                best_score = score
                best_contour = contour
                best_method = name
    
    if best_contour is not None and best_score >= 3:
        cv2.fillPoly(mask, [best_contour], 255)
        area = cv2.contourArea(best_contour)
        print(f"  Selected contour ({best_method}, score: {best_score}) with area: {area:.0f} pixels")
        
        # Refine the mask with advanced techniques
        blurred_for_refine = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = refine_mask_advanced(mask, gray, blurred_for_refine)
        return mask
    
    # Last resort: take largest valid contour
    for name, thresh_img in approaches:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
        
        all_contours = [(c, cv2.contourArea(c)) for c in contours if cv2.contourArea(c) >= min_area]
        if len(all_contours) == 0:
            continue
        
        all_contours.sort(key=lambda x: x[1], reverse=True)
        kidney_contour, area = all_contours[0]
        
        cv2.fillPoly(mask, [kidney_contour], 255)
        print(f"  Selected contour ({name}, fallback) with area: {area:.0f} pixels")
        blurred_for_refine = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = refine_mask_advanced(mask, gray, blurred_for_refine)
        return mask
    
    return mask


def compute_presence(mask: np.ndarray, area_threshold: float = 0.01) -> Tuple[bool, float]:
    """
    Compute whether kidney is present based on mask area.
    
    Args:
        mask: Binary mask (0 or 255)
        area_threshold: Minimum fraction of pixels that must be kidney (default: 0.01 = 1%)
        
    Returns:
        Tuple of (is_present: bool, fraction: float)
    """
    total_pixels = mask.size
    kidney_pixels = np.sum(mask > 127)
    fraction = kidney_pixels / total_pixels if total_pixels > 0 else 0.0
    
    is_present = fraction >= area_threshold
    
    return is_present, fraction


def overlay_mask(orig_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Create overlay image with kidney region highlighted.
    
    Args:
        orig_bgr: Original image in BGR format
        mask: Binary mask (0 or 255)
        alpha: Transparency factor for overlay (default: 0.4)
        
    Returns:
        Overlay image in BGR format
    """
    # Create a copy of the original
    overlay = orig_bgr.copy()
    
    # Create colored overlay (red for kidney region)
    colored_overlay = overlay.copy()
    colored_overlay[mask > 127] = [0, 0, 255]  # Red in BGR
    
    # Alpha blend
    result = cv2.addWeighted(overlay, 1.0 - alpha, colored_overlay, alpha, 0)
    
    # Draw contour outline in yellow for better visibility
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # Draw the largest contour outline
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(result, [largest_contour], -1, (0, 255, 255), 2)  # Yellow in BGR
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Segment kidney in ultrasound image using classical computer vision"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input ultrasound image"
    )
    parser.add_argument(
        "--out_overlay",
        type=str,
        default="kidney_overlay.png",
        help="Path to save overlay image (default: kidney_overlay.png)"
    )
    parser.add_argument(
        "--out_mask",
        type=str,
        default="kidney_mask.png",
        help="Path to save binary mask image (default: kidney_mask.png)"
    )
    parser.add_argument(
        "--presence_thresh",
        type=float,
        default=0.01,
        help="Minimum fraction of pixels for kidney presence (default: 0.01 = 1%%)"
    )
    parser.add_argument(
        "--canny_low",
        type=int,
        default=30,
        help="Lower threshold for Canny edge detection (default: 30)"
    )
    parser.add_argument(
        "--canny_high",
        type=int,
        default=80,
        help="Upper threshold for Canny edge detection (default: 80)"
    )
    
    args = parser.parse_args()
    
    # Load image
    print(f"Loading image: {args.image}")
    try:
        gray = load_gray(args.image)
        orig_bgr = cv2.imread(args.image)
        if orig_bgr is None:
            raise ValueError(f"Could not load BGR version of image: {args.image}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Image size: {gray.shape[1]}x{gray.shape[0]} pixels")
    
    # Enhance contrast
    print("Enhancing contrast...")
    enhanced = enhance_contrast(gray)
    
    # Find kidney mask
    print("Detecting kidney region...")
    mask = find_kidney_mask(
        enhanced,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        min_frac=0.01,
        max_frac=0.65
    )
    
    # Compute presence
    is_present, fraction = compute_presence(mask, args.presence_thresh)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Kidney present: {is_present}")
    print(f"Mask fraction: {fraction:.6f} ({fraction*100:.2f}%)")
    print("="*60)
    
    # Create overlay
    print("\nCreating overlay visualization...")
    overlay = overlay_mask(orig_bgr, mask)
    
    # Save outputs
    mask_path = Path(args.out_mask).absolute()
    overlay_path = Path(args.out_overlay).absolute()
    
    print(f"\nSaving outputs...")
    cv2.imwrite(str(mask_path), mask)
    print(f"  Mask saved to: {mask_path}")
    
    cv2.imwrite(str(overlay_path), overlay)
    print(f"  Overlay saved to: {overlay_path}")
    
    print("\n✅ Segmentation complete!")


if __name__ == "__main__":
    main()

