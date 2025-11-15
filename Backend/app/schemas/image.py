"""
Pydantic schemas for Image endpoints.

These schemas define the shape of data for API requests and responses
related to ultrasound images.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Request Schemas (what clients send to the API)
# ============================================================================

class ImageUploadMetadata(BaseModel):
    """
    Metadata accompanying an image upload.

    The actual file is sent as multipart/form-data.
    This schema represents additional metadata fields.

    Used by: POST /api/v1/sessions/{session_id}/images
    """

    # Note: The actual file upload is handled separately via FastAPI's UploadFile
    # This schema is for any additional metadata that might be sent with the upload

    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional description of this image",
        examples=["4-chamber cardiac view", "Initial scan attempt"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "description": "Cardiac 4-chamber view - attempt 1"
                }
            ]
        }
    }


# ============================================================================
# Response Schemas (what the API sends to clients)
# ============================================================================

class ImageResponse(BaseModel):
    """
    Schema for image data in API responses.

    This is what clients receive when they:
    - Upload an image (POST)
    - Get an image (GET)
    - List images (GET)

    Includes URL to retrieve the actual image file.
    """

    id: int = Field(
        description="Unique image identifier"
    )

    session_id: int = Field(
        description="ID of session this image belongs to"
    )

    filename: str = Field(
        description="Original filename"
    )

    file_size: int = Field(
        description="File size in bytes"
    )

    mime_type: str = Field(
        description="MIME type (image/png, image/jpeg, etc.)"
    )

    width: Optional[int] = Field(
        None,
        description="Image width in pixels"
    )

    height: Optional[int] = Field(
        None,
        description="Image height in pixels"
    )

    url: str = Field(
        description="URL to retrieve the image file"
    )

    thumbnail_url: Optional[str] = Field(
        None,
        description="URL to retrieve thumbnail (if available)"
    )

    created_at: datetime = Field(
        description="When the image was uploaded"
    )

    # Computed fields
    dimensions: Optional[str] = Field(
        None,
        description="Image dimensions as string (e.g., '800x600')"
    )

    file_size_formatted: Optional[str] = Field(
        None,
        description="Human-readable file size (e.g., '2.5 MB')"
    )

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": 42,
                    "session_id": 1,
                    "filename": "ultrasound_001.png",
                    "file_size": 1048576,
                    "mime_type": "image/png",
                    "width": 800,
                    "height": 600,
                    "url": "/api/v1/images/42/file",
                    "thumbnail_url": "/api/v1/images/42/thumbnail",
                    "created_at": "2024-11-15T12:30:00Z",
                    "dimensions": "800x600",
                    "file_size_formatted": "1.0 MB"
                }
            ]
        }
    }


class ImageUploadResponse(ImageResponse):
    """
    Response after successfully uploading an image.

    Extends ImageResponse with upload-specific information.

    Used by: POST /api/v1/sessions/{session_id}/images
    """

    message: str = Field(
        default="Image uploaded successfully",
        description="Success message"
    )

    model_config = {
        "from_attributes": True,
        "json_schema_extra": {
            "examples": [
                {
                    "id": 42,
                    "session_id": 1,
                    "filename": "ultrasound_001.png",
                    "file_size": 1048576,
                    "mime_type": "image/png",
                    "width": 800,
                    "height": 600,
                    "url": "/api/v1/images/42/file",
                    "thumbnail_url": None,
                    "created_at": "2024-11-15T12:30:00Z",
                    "dimensions": "800x600",
                    "file_size_formatted": "1.0 MB",
                    "message": "Image uploaded successfully"
                }
            ]
        }
    }


class ImageListResponse(BaseModel):
    """
    Paginated list of images.

    Used by: GET /api/v1/sessions/{session_id}/images
    """

    items: list[ImageResponse] = Field(
        description="List of images"
    )

    total: int = Field(
        description="Total number of images"
    )

    session_id: int = Field(
        description="Session ID these images belong to"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "items": [
                        {
                            "id": 42,
                            "session_id": 1,
                            "filename": "scan_001.png",
                            "file_size": 1048576,
                            "mime_type": "image/png",
                            "width": 800,
                            "height": 600,
                            "url": "/api/v1/images/42/file",
                            "thumbnail_url": None,
                            "created_at": "2024-11-15T12:30:00Z",
                            "dimensions": "800x600",
                            "file_size_formatted": "1.0 MB"
                        }
                    ],
                    "total": 5,
                    "session_id": 1
                }
            ]
        }
    }


# ============================================================================
# Validation Schemas
# ============================================================================

class ImageValidation(BaseModel):
    """
    Image validation rules used by the upload endpoint.

    Not sent in requests/responses, but used internally to validate uploads.
    """

    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Maximum file size in bytes"
    )

    allowed_mime_types: list[str] = Field(
        default=["image/png", "image/jpeg", "image/jpg"],
        description="Allowed MIME types"
    )

    min_width: int = Field(
        default=100,
        description="Minimum image width in pixels"
    )

    min_height: int = Field(
        default=100,
        description="Minimum image height in pixels"
    )

    max_width: int = Field(
        default=4096,
        description="Maximum image width in pixels"
    )

    max_height: int = Field(
        default=4096,
        description="Maximum image height in pixels"
    )

    @classmethod
    def validate_file_size(cls, size: int) -> bool:
        """Check if file size is within limits."""
        return 0 < size <= cls.model_fields['max_file_size'].default

    @classmethod
    def validate_mime_type(cls, mime_type: str) -> bool:
        """Check if MIME type is allowed."""
        return mime_type in cls.model_fields['allowed_mime_types'].default

    @classmethod
    def validate_dimensions(cls, width: int, height: int) -> bool:
        """Check if image dimensions are within limits."""
        min_w = cls.model_fields['min_width'].default
        max_w = cls.model_fields['max_width'].default
        min_h = cls.model_fields['min_height'].default
        max_h = cls.model_fields['max_height'].default

        return (min_w <= width <= max_w) and (min_h <= height <= max_h)
