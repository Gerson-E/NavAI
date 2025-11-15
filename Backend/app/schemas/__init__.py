"""
Pydantic schemas package.

This module exports all Pydantic schemas for API request/response validation.

Usage in API routes:
    from app.schemas import SessionCreate, SessionResponse
    from app.schemas import ImageResponse, ImageUploadResponse
    from app.schemas import ComparisonRequest, ComparisonResponse
"""

# Session schemas
from app.schemas.session import (
    SessionCreate,
    SessionUpdate,
    SessionResponse,
    SessionDetail,
    SessionListResponse,
)

# Image schemas
from app.schemas.image import (
    ImageUploadMetadata,
    ImageResponse,
    ImageUploadResponse,
    ImageListResponse,
    ImageValidation,
)

# Reference view schemas
from app.schemas.reference_view import (
    ReferenceViewCreate,
    ReferenceViewResponse,
    ReferenceViewListResponse,
    ReferenceViewDetail,
    ReferenceViewsByCategory,
)

# Comparison schemas (THE BRIDGE TO PERSON B)
from app.schemas.comparison import (
    ComparisonRequest,
    ComparisonResponse,
    ComparisonListResponse,
    AnalysisResultFromEngine,
    ComparisonStatistics,
)

# Classification schemas (MVP KIDNEY DETECTION)
from app.schemas.classification import (
    ClassificationRequest,
    ClassificationResponse,
    ClassificationResultFromEngine,
)


# Export all schemas
__all__ = [
    # Session schemas
    "SessionCreate",
    "SessionUpdate",
    "SessionResponse",
    "SessionDetail",
    "SessionListResponse",

    # Image schemas
    "ImageUploadMetadata",
    "ImageResponse",
    "ImageUploadResponse",
    "ImageListResponse",
    "ImageValidation",

    # Reference view schemas
    "ReferenceViewCreate",
    "ReferenceViewResponse",
    "ReferenceViewListResponse",
    "ReferenceViewDetail",
    "ReferenceViewsByCategory",

    # Comparison schemas
    "ComparisonRequest",
    "ComparisonResponse",
    "ComparisonListResponse",
    "AnalysisResultFromEngine",
    "ComparisonStatistics",

    # Classification schemas
    "ClassificationRequest",
    "ClassificationResponse",
    "ClassificationResultFromEngine",
]
