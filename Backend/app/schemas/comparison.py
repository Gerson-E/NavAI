"""
Pydantic schemas for Comparison/Analysis endpoints.

These schemas define the shape of data for image analysis requests and responses.
This is THE BRIDGE between your API layer and Person B's analysis engine.

The schemas here map directly to the ComparisonResult TypedDict from
app.analysis.interface.py.
"""

from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Request Schemas (what clients send to the API)
# ============================================================================

class ComparisonRequest(BaseModel):
    """
    Request to analyze an ultrasound image against a reference view.

    This is THE MAIN ENDPOINT for the application - where operators
    get real-time feedback on their probe positioning.

    Used by: POST /api/v1/check-position

    Flow:
    1. Client sends this request
    2. Your API fetches the image from storage
    3. Your API calls Person B's compare_to_reference()
    4. Your API saves the result to the database
    5. Your API returns ComparisonResponse to the client
    """

    session_id: int = Field(
        gt=0,
        description="ID of the current scan session"
    )

    image_id: int = Field(
        gt=0,
        description="ID of the ultrasound image to analyze"
    )

    reference_view_id: str = Field(
        min_length=1,
        max_length=50,
        description="ID of the reference view to compare against",
        examples=["cardiac_4chamber", "liver_standard", "kidney_longitudinal"]
    )

    @field_validator('reference_view_id')
    @classmethod
    def validate_reference_id(cls, v: str) -> str:
        """Ensure reference ID is properly formatted."""
        v = v.strip().lower()
        if not v:
            raise ValueError('Reference view ID cannot be empty')
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": 1,
                    "image_id": 42,
                    "reference_view_id": "cardiac_4chamber"
                }
            ]
        }
    }


# ============================================================================
# Response Schemas (what the API sends to clients)
# ============================================================================

class ComparisonResponse(BaseModel):
    """
    Result of analyzing an ultrasound image against a reference view.

    This schema maps DIRECTLY to the ComparisonResult TypedDict from
    app.analysis.interface.py (Person B's contract).

    This is what clients receive from POST /api/v1/check-position

    The frontend will display this to the ultrasound operator with:
    - Color coding based on verdict (green/yellow/red)
    - SSIM and NCC scores
    - Feedback message from Person B's analysis
    """

    # Database fields (not in Person B's result)
    id: int = Field(
        description="Unique comparison identifier (for history tracking)"
    )

    session_id: int = Field(
        description="Session this comparison belongs to"
    )

    image_id: int = Field(
        description="Image that was analyzed"
    )

    reference_view_id: str = Field(
        description="Reference view used for comparison"
    )

    # Analysis results (from Person B's engine - maps to ComparisonResult)
    ssim: float = Field(
        ge=0.0,
        le=1.0,
        description="Structural Similarity Index (0.0 to 1.0, higher is better)"
    )

    ncc: float = Field(
        ge=-1.0,
        le=1.0,
        description="Normalized Cross-Correlation (-1.0 to 1.0, higher is better)"
    )

    verdict: Literal["good", "borderline", "poor"] = Field(
        description="Overall assessment of probe positioning"
    )

    message: str = Field(
        description="Human-readable feedback for the operator",
        examples=[
            "Probe positioning looks good. SSIM: 0.82, NCC: 0.75",
            "Positioning needs minor adjustment. Try rotating probe slightly clockwise.",
            "Poor positioning detected. Reposition probe to match reference view."
        ]
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score of the analysis (0.0 to 1.0)"
    )

    # Metadata
    processing_time_ms: Optional[int] = Field(
        None,
        description="How long the analysis took in milliseconds"
    )

    created_at: datetime = Field(
        description="When the comparison was performed"
    )

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "session_id": 1,
                    "image_id": 42,
                    "reference_view_id": "cardiac_4chamber",
                    "ssim": 0.82,
                    "ncc": 0.75,
                    "verdict": "good",
                    "message": "Probe positioning looks good for cardiac 4-chamber view.",
                    "confidence": 0.88,
                    "processing_time_ms": 250,
                    "created_at": "2024-11-15T12:45:00Z"
                },
                {
                    "id": 2,
                    "session_id": 1,
                    "image_id": 43,
                    "reference_view_id": "cardiac_4chamber",
                    "ssim": 0.62,
                    "ncc": 0.58,
                    "verdict": "borderline",
                    "message": "Positioning needs minor adjustment. Try rotating probe slightly.",
                    "confidence": 0.72,
                    "processing_time_ms": 245,
                    "created_at": "2024-11-15T12:46:00Z"
                },
                {
                    "id": 3,
                    "session_id": 1,
                    "image_id": 44,
                    "reference_view_id": "cardiac_4chamber",
                    "ssim": 0.35,
                    "ncc": 0.28,
                    "verdict": "poor",
                    "message": "Poor positioning detected. Please reposition probe significantly.",
                    "confidence": 0.91,
                    "processing_time_ms": 238,
                    "created_at": "2024-11-15T12:47:00Z"
                }
            ]
        }
    }


class ComparisonListResponse(BaseModel):
    """
    List of comparisons for a session.

    Used by: GET /api/v1/sessions/{session_id}/comparisons

    Allows operators to review their positioning history.
    """

    items: list[ComparisonResponse] = Field(
        description="List of comparisons"
    )

    total: int = Field(
        description="Total number of comparisons"
    )

    session_id: int = Field(
        description="Session ID"
    )

    # Statistics
    average_ssim: Optional[float] = Field(
        None,
        description="Average SSIM score across all comparisons"
    )

    average_ncc: Optional[float] = Field(
        None,
        description="Average NCC score across all comparisons"
    )

    verdict_distribution: Optional[dict[str, int]] = Field(
        None,
        description="Count of each verdict type",
        examples=[{"good": 5, "borderline": 2, "poor": 1}]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "items": [
                        {
                            "id": 1,
                            "session_id": 1,
                            "image_id": 42,
                            "reference_view_id": "cardiac_4chamber",
                            "ssim": 0.82,
                            "ncc": 0.75,
                            "verdict": "good",
                            "message": "Positioning looks good",
                            "confidence": 0.88,
                            "processing_time_ms": 250,
                            "created_at": "2024-11-15T12:45:00Z"
                        }
                    ],
                    "total": 5,
                    "session_id": 1,
                    "average_ssim": 0.78,
                    "average_ncc": 0.71,
                    "verdict_distribution": {
                        "good": 3,
                        "borderline": 1,
                        "poor": 1
                    }
                }
            ]
        }
    }


# ============================================================================
# Helper Schemas for Person B Integration
# ============================================================================

class AnalysisResultFromEngine(BaseModel):
    """
    Schema matching Person B's ComparisonResult from interface.py.

    This is used internally to validate the result from Person B's
    compare_to_reference() function before saving to database.

    This ensures Person B's output matches the contract.
    """

    ssim: float = Field(
        ge=0.0,
        le=1.0,
        description="Structural Similarity Index"
    )

    ncc: float = Field(
        ge=-1.0,
        le=1.0,
        description="Normalized Cross-Correlation"
    )

    verdict: Literal["good", "borderline", "poor"] = Field(
        description="Assessment verdict"
    )

    message: str = Field(
        min_length=1,
        description="Feedback message"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ssim": 0.82,
                    "ncc": 0.75,
                    "verdict": "good",
                    "message": "Probe positioning looks good",
                    "confidence": 0.88
                }
            ]
        }
    }


# ============================================================================
# Statistics Schemas
# ============================================================================

class ComparisonStatistics(BaseModel):
    """
    Statistics about comparisons for a session or user.

    Used by: GET /api/v1/sessions/{id}/statistics
    """

    total_comparisons: int = Field(
        description="Total number of comparisons"
    )

    verdict_counts: dict[str, int] = Field(
        description="Count of each verdict type"
    )

    average_scores: dict[str, float] = Field(
        description="Average SSIM and NCC scores"
    )

    average_processing_time_ms: Optional[float] = Field(
        None,
        description="Average processing time"
    )

    improvement_trend: Optional[str] = Field(
        None,
        description="Whether scores are improving, declining, or stable",
        examples=["improving", "declining", "stable"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "total_comparisons": 15,
                    "verdict_counts": {
                        "good": 8,
                        "borderline": 5,
                        "poor": 2
                    },
                    "average_scores": {
                        "ssim": 0.75,
                        "ncc": 0.68
                    },
                    "average_processing_time_ms": 245.3,
                    "improvement_trend": "improving"
                }
            ]
        }
    }
