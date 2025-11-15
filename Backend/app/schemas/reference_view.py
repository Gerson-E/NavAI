"""
Pydantic schemas for Reference View endpoints.

These schemas define the shape of data for API requests and responses
related to reference ultrasound views.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Request Schemas (what clients send to the API)
# ============================================================================

class ReferenceViewCreate(BaseModel):
    """
    Schema for creating a new reference view.

    Used by: POST /api/v1/reference-views (admin only, future)

    Most reference views will be seeded initially, but this allows
    admins to add new ones later.
    """

    id: str = Field(
        min_length=1,
        max_length=50,
        description="Unique identifier (e.g., 'cardiac_4chamber')",
        examples=["cardiac_4chamber", "liver_standard"]
    )

    name: str = Field(
        min_length=1,
        max_length=255,
        description="Display name",
        examples=["Cardiac 4-Chamber View", "Standard Liver View"]
    )

    description: Optional[str] = Field(
        None,
        max_length=2000,
        description="Detailed description of this view",
        examples=[
            "Standard apical 4-chamber cardiac view showing all four chambers of the heart"
        ]
    )

    category: str = Field(
        min_length=1,
        max_length=50,
        description="Anatomical category",
        examples=["cardiac", "abdominal", "vascular"]
    )

    sort_order: int = Field(
        default=0,
        ge=0,
        description="Display order (lower numbers first)"
    )

    @field_validator('id')
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Ensure ID uses snake_case format."""
        if not v.replace('_', '').isalnum():
            raise ValueError('ID must contain only letters, numbers, and underscores')
        if v != v.lower():
            raise ValueError('ID must be lowercase')
        return v

    @field_validator('category')
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Validate category against allowed values."""
        allowed = ['cardiac', 'abdominal', 'vascular', 'obstetric', 'musculoskeletal', 'other']
        if v.lower() not in allowed:
            raise ValueError(f'Category must be one of: {", ".join(allowed)}')
        return v.lower()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "cardiac_4chamber",
                    "name": "Cardiac 4-Chamber View",
                    "description": "Standard apical 4-chamber view",
                    "category": "cardiac",
                    "sort_order": 1
                }
            ]
        }
    }


# ============================================================================
# Response Schemas (what the API sends to clients)
# ============================================================================

class ReferenceViewResponse(BaseModel):
    """
    Schema for reference view data in API responses.

    This is what clients receive when they:
    - List reference views (GET)
    - Get a specific reference view (GET)
    - Create a reference view (POST) - admin only

    Includes URLs to retrieve the reference image and thumbnail.
    """

    id: str = Field(
        description="Unique reference view identifier"
    )

    name: str = Field(
        description="Display name"
    )

    description: Optional[str] = Field(
        None,
        description="Detailed description"
    )

    category: str = Field(
        description="Anatomical category"
    )

    image_url: str = Field(
        description="URL to retrieve the reference image"
    )

    thumbnail_url: Optional[str] = Field(
        None,
        description="URL to retrieve thumbnail"
    )

    is_active: bool = Field(
        description="Whether this reference view is available"
    )

    sort_order: int = Field(
        description="Display order"
    )

    created_at: datetime = Field(
        description="When this reference was added"
    )

    # Statistics (optional, computed by API)
    usage_count: Optional[int] = Field(
        None,
        description="Number of times this reference has been used in comparisons"
    )

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": "cardiac_4chamber",
                    "name": "Cardiac 4-Chamber View",
                    "description": "Standard apical 4-chamber cardiac view",
                    "category": "cardiac",
                    "image_url": "/api/v1/reference-views/cardiac_4chamber/image",
                    "thumbnail_url": "/api/v1/reference-views/cardiac_4chamber/thumbnail",
                    "is_active": True,
                    "sort_order": 1,
                    "created_at": "2024-11-15T10:00:00Z",
                    "usage_count": 42
                }
            ]
        }
    }


class ReferenceViewListResponse(BaseModel):
    """
    List of available reference views.

    Used by: GET /api/v1/reference-views

    Returns all active reference views, grouped by category.
    """

    items: list[ReferenceViewResponse] = Field(
        description="List of reference views"
    )

    total: int = Field(
        description="Total number of reference views"
    )

    categories: list[str] = Field(
        description="List of available categories"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "items": [
                        {
                            "id": "cardiac_4chamber",
                            "name": "Cardiac 4-Chamber View",
                            "description": "Standard apical 4-chamber view",
                            "category": "cardiac",
                            "image_url": "/api/v1/reference-views/cardiac_4chamber/image",
                            "thumbnail_url": None,
                            "is_active": True,
                            "sort_order": 1,
                            "created_at": "2024-11-15T10:00:00Z",
                            "usage_count": 42
                        },
                        {
                            "id": "liver_standard",
                            "name": "Standard Liver View",
                            "description": "Standard liver parenchyma view",
                            "category": "abdominal",
                            "image_url": "/api/v1/reference-views/liver_standard/image",
                            "thumbnail_url": None,
                            "is_active": True,
                            "sort_order": 10,
                            "created_at": "2024-11-15T10:00:00Z",
                            "usage_count": 28
                        }
                    ],
                    "total": 2,
                    "categories": ["cardiac", "abdominal"]
                }
            ]
        }
    }


class ReferenceViewDetail(ReferenceViewResponse):
    """
    Detailed reference view with additional metadata.

    Used by: GET /api/v1/reference-views/{id}

    Includes statistics and usage information.
    """

    # Additional statistics
    recent_comparisons: Optional[int] = Field(
        None,
        description="Number of comparisons in the last 30 days"
    )

    average_ssim: Optional[float] = Field(
        None,
        description="Average SSIM score across all comparisons"
    )

    average_ncc: Optional[float] = Field(
        None,
        description="Average NCC score across all comparisons"
    )

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": "cardiac_4chamber",
                    "name": "Cardiac 4-Chamber View",
                    "description": "Standard apical 4-chamber cardiac view",
                    "category": "cardiac",
                    "image_url": "/api/v1/reference-views/cardiac_4chamber/image",
                    "thumbnail_url": None,
                    "is_active": True,
                    "sort_order": 1,
                    "created_at": "2024-11-15T10:00:00Z",
                    "usage_count": 42,
                    "recent_comparisons": 12,
                    "average_ssim": 0.78,
                    "average_ncc": 0.72
                }
            ]
        }
    }


# ============================================================================
# Helper Schemas
# ============================================================================

class ReferenceViewsByCategory(BaseModel):
    """
    Reference views grouped by category.

    Used by: GET /api/v1/reference-views?group_by=category
    """

    category: str = Field(
        description="Category name"
    )

    views: list[ReferenceViewResponse] = Field(
        description="Reference views in this category"
    )

    count: int = Field(
        description="Number of views in this category"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "category": "cardiac",
                    "views": [
                        {
                            "id": "cardiac_4chamber",
                            "name": "Cardiac 4-Chamber View",
                            "description": None,
                            "category": "cardiac",
                            "image_url": "/api/v1/reference-views/cardiac_4chamber/image",
                            "thumbnail_url": None,
                            "is_active": True,
                            "sort_order": 1,
                            "created_at": "2024-11-15T10:00:00Z",
                            "usage_count": 42
                        }
                    ],
                    "count": 1
                }
            ]
        }
    }
