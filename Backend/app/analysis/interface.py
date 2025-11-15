"""
Interface contract between API layer (Person A) and Analysis Engine (Person B).

This file defines the contract that both teams must follow.
DO NOT modify this file without agreement from both Person A and Person B.

Last updated: 2025-11-15
"""

from typing import TypedDict, Literal


class ComparisonResult(TypedDict):
    """
    Result of comparing a current ultrasound image to a reference view.

    Fields:
        ssim: Structural Similarity Index (0.0 to 1.0, higher is better)
        ncc: Normalized Cross-Correlation (-1.0 to 1.0, higher is better)
        verdict: Overall assessment of the probe positioning
        message: Human-readable feedback message for the operator
        confidence: Confidence score of the analysis (0.0 to 1.0)
    """
    ssim: float
    ncc: float
    verdict: Literal["good", "borderline", "poor"]
    message: str
    confidence: float


def compare_to_reference(current_img_path: str, ref_id: str) -> ComparisonResult:
    """
    Compare a current ultrasound image to a reference view.

    Args:
        current_img_path: Absolute path to the current ultrasound image file
        ref_id: Identifier for the reference view to compare against
                (e.g., "cardiac_4chamber", "liver_standard", etc.)

    Returns:
        ComparisonResult dictionary with similarity metrics and verdict

    Raises:
        FileNotFoundError: If current_img_path does not exist
        ValueError: If ref_id is not recognized or image is invalid/corrupted
        RuntimeError: If analysis fails due to processing error

    Example:
        >>> result = compare_to_reference("/path/to/scan.png", "cardiac_4chamber")
        >>> print(result["verdict"])
        'good'
        >>> print(result["ssim"])
        0.82
    """
    raise NotImplementedError("This function must be implemented by Person B")
