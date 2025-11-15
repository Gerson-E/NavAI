"""
Image Analysis Engine - Person B's implementation.

This is currently a STUB implementation that returns dummy data.
Person B will replace this with the real computer vision analysis.

Person A: You can import and use this for testing your API while Person B builds the real implementation.
"""

import os
from app.analysis.interface import ComparisonResult


def compare_to_reference(current_img_path: str, ref_id: str) -> ComparisonResult:
    """
    STUB IMPLEMENTATION - Returns dummy data for testing.

    Person B will replace this with real image analysis logic including:
    - Image preprocessing
    - SSIM calculation
    - NCC calculation
    - Verdict determination based on thresholds

    Args:
        current_img_path: Absolute path to the current ultrasound image file
        ref_id: Identifier for the reference view to compare against

    Returns:
        ComparisonResult dictionary with similarity metrics and verdict

    Raises:
        FileNotFoundError: If current_img_path does not exist
        ValueError: If ref_id is not recognized
    """
    # Basic validation that Person A can rely on
    if not os.path.exists(current_img_path):
        raise FileNotFoundError(f"Image file not found: {current_img_path}")

    # TODO: Person B - Add ref_id validation against known reference views
    valid_ref_ids = ["cardiac_4chamber", "liver_standard", "kidney_longitudinal"]
    if ref_id not in valid_ref_ids:
        raise ValueError(f"Unknown reference ID: {ref_id}. Valid IDs: {valid_ref_ids}")

    # STUB: Return dummy data with reasonable values
    # Person B will replace everything below with real analysis
    return ComparisonResult(
        ssim=0.78,
        ncc=0.72,
        verdict="good",
        message=f"[STUB] Probe positioning looks good for {ref_id}. This is dummy data.",
        confidence=0.85
    )


# Person B: Add your additional helper functions below
# Examples:
# - def load_image(path: str) -> np.ndarray
# - def preprocess_ultrasound(img: np.ndarray) -> np.ndarray
# - def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float
# - def compute_ncc(img1: np.ndarray, img2: np.ndarray) -> float
# - def interpret_scores(ssim: float, ncc: float) -> tuple[str, str, float]
