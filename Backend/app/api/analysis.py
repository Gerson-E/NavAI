"""
Analysis API endpoints.

This module provides the main feature: checking ultrasound probe positioning
by comparing images to reference views using Person B's analysis engine.

This is THE BRIDGE between your API layer and Person B's CV analysis.

Endpoints:
    POST /check-position - Analyze image positioning
    GET  /sessions/{id}/comparisons - List session comparisons
    GET  /comparisons/{id} - Get comparison details
"""

import time
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session as DBSession

from app.core.database import get_db
from app.core.storage import get_image_path
from app.models import Session, Image, Comparison, ReferenceView
from app.schemas import (
    ComparisonRequest,
    ComparisonResponse,
    ComparisonListResponse,
    AnalysisResultFromEngine,
)

# Import Person B's analysis engine
from app.analysis.engine import compare_to_reference


router = APIRouter()


# ============================================================================
# Helper Functions
# ============================================================================

def get_session_or_404(db: DBSession, session_id: int) -> Session:
    """Get session by ID or raise 404."""
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    return session


def get_image_or_404(db: DBSession, image_id: int) -> Image:
    """Get image by ID or raise 404."""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found"
        )
    return image


def get_reference_view_or_404(db: DBSession, ref_id: str) -> ReferenceView:
    """Get reference view by ID or raise 404."""
    ref_view = db.query(ReferenceView).filter(ReferenceView.id == ref_id).first()
    if not ref_view:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference view '{ref_id}' not found"
        )
    return ref_view


def get_comparison_or_404(db: DBSession, comparison_id: int) -> Comparison:
    """Get comparison by ID or raise 404."""
    comparison = db.query(Comparison).filter(Comparison.id == comparison_id).first()
    if not comparison:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Comparison {comparison_id} not found"
        )
    return comparison


# ============================================================================
# Check Position Endpoint (THE MAIN FEATURE)
# ============================================================================

@router.post(
    "/check-position",
    response_model=ComparisonResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Check probe positioning",
    description="Analyze ultrasound image and compare to reference view"
)
def check_position(
    request: ComparisonRequest,
    db: DBSession = Depends(get_db)
) -> ComparisonResponse:
    """
    Check ultrasound probe positioning by comparing image to reference view.

    **This is THE MAIN FEATURE of the application.**

    **Process:**
    1. Validates session, image, and reference view exist
    2. Gets image file path from storage
    3. Calls Person B's analysis engine (compare_to_reference)
    4. Saves comparison result to database
    5. Returns positioning feedback to operator

    **Person B Integration:**
    - Calls: `compare_to_reference(image_path, reference_view_id)`
    - Receives: `{"ssim": float, "ncc": float, "verdict": str, "message": str, "confidence": float}`
    - Validates result matches interface contract
    - Stores result for historical tracking

    **Response:**
    - SSIM score (0.0 to 1.0, higher is better)
    - NCC score (-1.0 to 1.0, higher is better)
    - Verdict: "good", "borderline", or "poor"
    - Human-readable message for operator
    - Confidence score (0.0 to 1.0)

    **Use case:**
    Operator uploads an ultrasound image and gets immediate feedback on
    whether their probe positioning matches the reference view.

    **Color coding in frontend:**
    - "good" → Green (SSIM > 0.75)
    - "borderline" → Yellow (0.5 < SSIM ≤ 0.75)
    - "poor" → Red (SSIM ≤ 0.5)
    """
    # ========================================================================
    # 1. Validate all required entities exist
    # ========================================================================

    # Verify session exists
    session = get_session_or_404(db, request.session_id)

    # Verify image exists and belongs to this session
    image = get_image_or_404(db, request.image_id)
    if image.session_id != request.session_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image {request.image_id} does not belong to session {request.session_id}"
        )

    # Verify reference view exists
    # Note: For now we're not checking the database, just validating with Person B
    # Future: When reference views are in DB, use get_reference_view_or_404()

    # ========================================================================
    # 2. Get image file path from storage
    # ========================================================================

    try:
        image_path = get_image_path(image.storage_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found on disk: {image.storage_path}"
        )

    # ========================================================================
    # 3. Call Person B's analysis engine
    # ========================================================================

    start_time = time.time()

    try:
        # THE BRIDGE TO PERSON B
        # This calls Person B's compare_to_reference() function
        result = compare_to_reference(
            current_img_path=str(image_path),
            ref_id=request.reference_view_id
        )

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference image not found: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Analysis validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis engine error: {str(e)}"
        )

    # ========================================================================
    # 4. Validate result from Person B matches interface contract
    # ========================================================================

    try:
        # This validates that Person B's output matches the expected format
        validated_result = AnalysisResultFromEngine(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis result validation failed: {str(e)}"
        )

    # ========================================================================
    # 5. Save comparison result to database
    # ========================================================================

    # Use the factory method from Comparison model
    comparison = Comparison.from_analysis_result(
        session_id=request.session_id,
        image_id=request.image_id,
        reference_view_id=request.reference_view_id,
        result=result,
        processing_time_ms=processing_time_ms
    )

    db.add(comparison)
    db.commit()
    db.refresh(comparison)

    # ========================================================================
    # 6. Return response to operator
    # ========================================================================

    return ComparisonResponse(
        id=comparison.id,
        session_id=comparison.session_id,
        image_id=comparison.image_id,
        reference_view_id=comparison.reference_view_id,
        ssim=comparison.ssim,
        ncc=comparison.ncc,
        verdict=comparison.verdict.value,
        message=comparison.message,
        confidence=comparison.confidence,
        processing_time_ms=comparison.processing_time_ms,
        created_at=comparison.created_at
    )


# ============================================================================
# List Comparisons
# ============================================================================

@router.get(
    "/sessions/{session_id}/comparisons",
    response_model=ComparisonListResponse,
    summary="List session comparisons",
    description="Get all positioning checks for a session"
)
def list_session_comparisons(
    session_id: int,
    db: DBSession = Depends(get_db)
) -> ComparisonListResponse:
    """
    Get all positioning comparisons for a session.

    Shows the history of all positioning checks performed during a session.
    Useful for tracking operator improvement and reviewing past analyses.

    **Includes:**
    - All comparison results
    - Average SSIM and NCC scores
    - Verdict distribution (how many good/borderline/poor)

    **Ordering:**
    Comparisons are ordered by creation time (newest first).

    **Use case:**
    Display session history showing how positioning improved over time.
    """
    # Verify session exists
    get_session_or_404(db, session_id)

    # Get all comparisons for this session
    comparisons = (
        db.query(Comparison)
        .filter(Comparison.session_id == session_id)
        .order_by(Comparison.created_at.desc())
        .all()
    )

    # Build comparison responses
    comparison_responses = [
        ComparisonResponse(
            id=comp.id,
            session_id=comp.session_id,
            image_id=comp.image_id,
            reference_view_id=comp.reference_view_id,
            ssim=comp.ssim,
            ncc=comp.ncc,
            verdict=comp.verdict.value,
            message=comp.message,
            confidence=comp.confidence,
            processing_time_ms=comp.processing_time_ms,
            created_at=comp.created_at
        )
        for comp in comparisons
    ]

    # Calculate statistics
    if comparisons:
        avg_ssim = sum(c.ssim for c in comparisons) / len(comparisons)
        avg_ncc = sum(c.ncc for c in comparisons) / len(comparisons)

        # Count verdicts
        verdict_dist = {"good": 0, "borderline": 0, "poor": 0}
        for comp in comparisons:
            verdict_dist[comp.verdict.value] += 1
    else:
        avg_ssim = None
        avg_ncc = None
        verdict_dist = None

    return ComparisonListResponse(
        items=comparison_responses,
        total=len(comparisons),
        session_id=session_id,
        average_ssim=avg_ssim,
        average_ncc=avg_ncc,
        verdict_distribution=verdict_dist
    )


# ============================================================================
# Get Comparison Details
# ============================================================================

@router.get(
    "/comparisons/{comparison_id}",
    response_model=ComparisonResponse,
    summary="Get comparison details",
    description="Get detailed information about a specific positioning check"
)
def get_comparison(
    comparison_id: int,
    db: DBSession = Depends(get_db)
) -> ComparisonResponse:
    """
    Get detailed information about a specific positioning comparison.

    **Use case:**
    Review a specific positioning check result, including all scores and feedback.
    """
    comparison = get_comparison_or_404(db, comparison_id)

    return ComparisonResponse(
        id=comparison.id,
        session_id=comparison.session_id,
        image_id=comparison.image_id,
        reference_view_id=comparison.reference_view_id,
        ssim=comparison.ssim,
        ncc=comparison.ncc,
        verdict=comparison.verdict.value,
        message=comparison.message,
        confidence=comparison.confidence,
        processing_time_ms=comparison.processing_time_ms,
        created_at=comparison.created_at
    )
