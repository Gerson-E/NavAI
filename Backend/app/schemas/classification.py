"""
Pydantic schemas for Classification/Organ Detection endpoints.

These schemas define the shape of data for organ classification requests and responses.
This is the MVP FEATURE for kidney detection.

The schemas here map directly to the ClassificationResult TypedDict from
app.analysis.interface.py.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# Request Schemas (what clients send to the API)
# ============================================================================

class ClassificationRequest(BaseModel):
    """
    Request to classify what organ is shown in an ultrasound image.

    This is THE MVP FEATURE - simple kidney detection.

    Used by: POST /api/v1/classify-organ

    Flow:
    1. Client sends this request with session_id and image_id
    2. Your API fetches the image from storage
    3. Your API calls Person B's classify_organ()
    4. Your API returns ClassificationResponse to the client
    """

    session_id: int = Field(
        gt=0,
        description="ID of the current scan session"
    )

    image_id: int = Field(
        gt=0,
        description="ID of the ultrasound image to classify"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "session_id": 1,
                    "image_id": 42
                }
            ]
        }
    }


# ============================================================================
# Response Schemas (what the API sends to clients)
# ============================================================================

class ClassificationResponse(BaseModel):
    """
    Result of classifying what organ is shown in an ultrasound image.

    This schema maps DIRECTLY to the ClassificationResult TypedDict from
    app.analysis.interface.py (Person B's contract).

    This is what clients receive from POST /api/v1/classify-organ

    The frontend will display this to the ultrasound operator with:
    - Detected organ type
    - Whether it's a kidney (for MVP)
    - Confidence score
    - Feedback message
    """

    # Database fields
    id: int = Field(
        description="Unique classification identifier (for history tracking)"
    )

    session_id: int = Field(
        description="Session this classification belongs to"
    )

    image_id: int = Field(
        description="Image that was classified"
    )

    # Analysis results (from Person B's engine - maps to ClassificationResult)
    detected_organ: str = Field(
        description="The identified organ or anatomy",
        examples=["kidney", "liver", "heart", "bladder", "unknown"]
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score of the classification (0.0 to 1.0)"
    )

    is_kidney: bool = Field(
        description="Boolean flag indicating if the image shows a kidney (MVP feature)"
    )

    message: str = Field(
        description="Human-readable feedback for the operator",
        examples=[
            "Kidney detected with high confidence",
            "This appears to be a kidney",
            "No kidney detected - appears to be liver",
            "Unable to identify organ clearly"
        ]
    )

    # Metadata
    processing_time_ms: Optional[int] = Field(
        None,
        description="How long the classification took in milliseconds"
    )

    created_at: datetime = Field(
        description="When the classification was performed"
    )

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "session_id": 1,
                    "image_id": 42,
                    "detected_organ": "kidney",
                    "confidence": 0.92,
                    "is_kidney": True,
                    "message": "Kidney detected with high confidence",
                    "processing_time_ms": 180,
                    "created_at": "2024-11-15T12:45:00Z"
                },
                {
                    "id": 2,
                    "session_id": 1,
                    "image_id": 43,
                    "detected_organ": "liver",
                    "confidence": 0.88,
                    "is_kidney": False,
                    "message": "No kidney detected - appears to be liver",
                    "processing_time_ms": 175,
                    "created_at": "2024-11-15T12:46:00Z"
                },
                {
                    "id": 3,
                    "session_id": 1,
                    "image_id": 44,
                    "detected_organ": "unknown",
                    "confidence": 0.45,
                    "is_kidney": False,
                    "message": "Unable to identify organ clearly - please reposition probe",
                    "processing_time_ms": 185,
                    "created_at": "2024-11-15T12:47:00Z"
                }
            ]
        }
    }


# ============================================================================
# Helper Schemas for Person B Integration
# ============================================================================

class ClassificationResultFromEngine(BaseModel):
    """
    Schema matching Person B's ClassificationResult from interface.py.

    This is used internally to validate the result from Person B's
    classify_organ() function before saving to database.

    This ensures Person B's output matches the contract.
    """

    detected_organ: str = Field(
        min_length=1,
        description="Detected organ type"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score"
    )

    is_kidney: bool = Field(
        description="Whether image shows a kidney"
    )

    message: str = Field(
        min_length=1,
        description="Feedback message"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detected_organ": "kidney",
                    "confidence": 0.92,
                    "is_kidney": True,
                    "message": "Kidney detected with high confidence"
                }
            ]
        }
    }
