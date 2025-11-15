"""
Image API endpoints.

This module provides endpoints for uploading, retrieving, and managing
ultrasound images.

Endpoints:
    POST   /sessions/{session_id}/images  - Upload image
    GET    /images/{image_id}             - Get image metadata
    GET    /images/{image_id}/file        - Download image file
    DELETE /images/{image_id}             - Delete image
    GET    /sessions/{session_id}/images  - List session images
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.storage import (
    save_uploaded_image,
    get_image_path,
    delete_image as storage_delete_image,
)
from app.core.config import settings
from app.models import Image, Session as SessionModel
from app.schemas import ImageResponse, ImageUploadResponse, ImageListResponse


router = APIRouter()


# ============================================================================
# Helper Functions
# ============================================================================

def get_session_or_404(db: Session, session_id: int) -> SessionModel:
    """
    Get session by ID or raise 404.

    Args:
        db: Database session
        session_id: Session ID

    Returns:
        SessionModel: The session

    Raises:
        HTTPException: 404 if session not found
    """
    session = db.query(SessionModel).filter(SessionModel.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    return session


def get_image_or_404(db: Session, image_id: int) -> Image:
    """
    Get image by ID or raise 404.

    Args:
        db: Database session
        image_id: Image ID

    Returns:
        Image: The image

    Raises:
        HTTPException: 404 if image not found
    """
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found"
        )
    return image


def build_image_response(image: Image, base_url: str = "") -> dict:
    """
    Build ImageResponse dict from Image model.

    Adds computed fields like URL, dimensions, file_size_formatted.

    Args:
        image: Image model instance
        base_url: Base URL for constructing image URLs

    Returns:
        dict: ImageResponse-compatible dict
    """
    return {
        "id": image.id,
        "session_id": image.session_id,
        "filename": image.filename,
        "file_size": image.file_size,
        "mime_type": image.mime_type,
        "width": image.width,
        "height": image.height,
        "url": f"{base_url}/api/v1/images/{image.id}/file",
        "thumbnail_url": None,  # TODO: Generate thumbnails in future
        "created_at": image.created_at,
        "dimensions": image.dimensions,
        "file_size_formatted": image.format_file_size(),
    }


# ============================================================================
# Upload Endpoint
# ============================================================================

@router.post(
    "/sessions/{session_id}/images",
    response_model=ImageUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload ultrasound image",
    description="Upload a new ultrasound image to a session"
)
async def upload_image(
    session_id: int,
    file: UploadFile = File(..., description="Image file (PNG or JPEG)"),
    db: Session = Depends(get_db)
) -> ImageUploadResponse:
    """
    Upload an ultrasound image to a session.

    **Process:**
    1. Validates session exists
    2. Validates file type (PNG, JPEG)
    3. Saves file to storage (organized by session)
    4. Extracts image metadata (dimensions, size)
    5. Creates database record
    6. Returns image metadata with URL

    **File Requirements:**
    - Format: PNG or JPEG
    - Max size: 10 MB (configured in settings)
    - Must be a valid image file

    **Storage:**
    - Stored in: `media/sessions/{session_id}/`
    - Filename: Timestamped and hashed for uniqueness
    - Example: `20241115_131139_a1b2c3_ultrasound.png`

    **Returns:**
    - Image metadata including ID, URL, dimensions, file size
    """
    # Verify session exists
    get_session_or_404(db, session_id)

    # Validate file type
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid file type: {file.content_type}. "
                f"Allowed types: {', '.join(settings.ALLOWED_IMAGE_TYPES)}"
            )
        )

    # Check file size (before reading entire file)
    # Note: This requires the client to send Content-Length header
    if hasattr(file, 'size') and file.size:
        if file.size > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"File too large: {file.size} bytes. "
                    f"Maximum allowed: {settings.MAX_UPLOAD_SIZE} bytes"
                )
            )

    try:
        # Save file to storage
        storage_path, metadata = await save_uploaded_image(file, session_id)

    except ValueError as e:
        # Validation error from storage module
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save image: {str(e)}"
        )

    # Create database record
    image = Image(
        session_id=session_id,
        filename=file.filename or "upload.png",
        storage_path=storage_path,
        file_size=metadata["file_size"],
        mime_type=metadata["mime_type"],
        width=metadata["width"],
        height=metadata["height"],
    )

    db.add(image)
    db.commit()
    db.refresh(image)

    # Build response
    response_data = build_image_response(image)
    response_data["message"] = "Image uploaded successfully"

    return ImageUploadResponse(**response_data)


# ============================================================================
# Retrieve Endpoints
# ============================================================================

@router.get(
    "/images/{image_id}",
    response_model=ImageResponse,
    summary="Get image metadata",
    description="Get metadata for a specific image"
)
def get_image_metadata(
    image_id: int,
    db: Session = Depends(get_db)
) -> ImageResponse:
    """
    Get metadata for a specific image.

    Returns image information including:
    - Filename, size, dimensions
    - Upload timestamp
    - URL to download the file
    - MIME type

    Does NOT return the actual image file - use GET /images/{id}/file for that.
    """
    image = get_image_or_404(db, image_id)
    response_data = build_image_response(image)
    return ImageResponse(**response_data)


@router.get(
    "/images/{image_id}/file",
    response_class=FileResponse,
    summary="Download image file",
    description="Download the actual image file"
)
def download_image_file(
    image_id: int,
    db: Session = Depends(get_db)
) -> FileResponse:
    """
    Download the actual image file.

    Returns the image file with appropriate headers for browser display
    or download.

    **Headers:**
    - Content-Type: image/png or image/jpeg
    - Content-Disposition: inline (displays in browser)
    """
    # Get image metadata from database
    image = get_image_or_404(db, image_id)

    # Get file path from storage
    try:
        file_path = get_image_path(image.storage_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image file not found on disk: {image.storage_path}"
        )

    # Return file
    return FileResponse(
        path=str(file_path),
        media_type=image.mime_type,
        filename=image.filename,
        headers={
            "Content-Disposition": f'inline; filename="{image.filename}"'
        }
    )


@router.get(
    "/sessions/{session_id}/images",
    response_model=ImageListResponse,
    summary="List session images",
    description="Get all images for a session"
)
def list_session_images(
    session_id: int,
    db: Session = Depends(get_db)
) -> ImageListResponse:
    """
    Get all images uploaded to a session.

    Returns a list of all images with their metadata.
    Images are ordered by upload time (newest first).

    **Use case:**
    Display all images captured during an ultrasound session.
    """
    # Verify session exists
    get_session_or_404(db, session_id)

    # Get all images for this session
    images = (
        db.query(Image)
        .filter(Image.session_id == session_id)
        .order_by(Image.created_at.desc())
        .all()
    )

    # Build response
    image_responses = [
        ImageResponse(**build_image_response(img))
        for img in images
    ]

    return ImageListResponse(
        items=image_responses,
        total=len(images),
        session_id=session_id
    )


# ============================================================================
# Delete Endpoint
# ============================================================================

@router.delete(
    "/images/{image_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete image",
    description="Delete an image and its file"
)
def delete_image(
    image_id: int,
    db: Session = Depends(get_db)
) -> None:
    """
    Delete an image.

    This will:
    1. Delete the image file from storage
    2. Delete the database record
    3. CASCADE delete any comparisons using this image

    **Warning:** This action cannot be undone.
    """
    # Get image
    image = get_image_or_404(db, image_id)

    # Delete file from storage
    try:
        storage_delete_image(image.storage_path)
    except Exception as e:
        # Log error but continue with database deletion
        # (file might already be deleted manually)
        print(f"Warning: Failed to delete file {image.storage_path}: {e}")

    # Delete database record (will CASCADE delete comparisons)
    db.delete(image)
    db.commit()

    # Return 204 No Content (FastAPI handles this automatically)
    return None
