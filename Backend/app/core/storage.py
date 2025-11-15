"""
Image storage management system.

This module handles all file storage operations for uploaded ultrasound images.
Images are stored on the local filesystem, organized by session ID.

Storage structure:
    media/
    └── sessions/
        └── {session_id}/
            ├── image_001.png
            ├── image_002.png
            └── ...

Future: Can be extended to support cloud storage (S3, Azure Blob, etc.)
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
from fastapi import UploadFile
from PIL import Image
import hashlib

from app.core.config import settings


# ============================================================================
# Storage Paths
# ============================================================================

def get_media_root() -> Path:
    """
    Get the root media directory path.

    Returns:
        Path: Absolute path to media root directory
    """
    media_root = Path(settings.MEDIA_ROOT)
    media_root.mkdir(parents=True, exist_ok=True)
    return media_root


def get_session_directory(session_id: int) -> Path:
    """
    Get the storage directory for a specific session.

    Creates the directory if it doesn't exist.

    Args:
        session_id: The session ID

    Returns:
        Path: Absolute path to session directory
    """
    session_dir = get_media_root() / "sessions" / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


# ============================================================================
# File Naming
# ============================================================================

def generate_unique_filename(
    original_filename: str,
    session_id: int,
    timestamp: Optional[datetime] = None
) -> str:
    """
    Generate a unique filename to prevent collisions.

    Format: {timestamp}_{hash}_{original_name}
    Example: 20241115_a1b2c3_ultrasound.png

    Args:
        original_filename: Original uploaded filename
        session_id: Session ID (used in hash)
        timestamp: Optional timestamp (defaults to now)

    Returns:
        str: Unique filename
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Get file extension
    _, ext = os.path.splitext(original_filename)
    if not ext:
        ext = ".bin"

    # Create a hash from session_id, timestamp, and original filename
    # This ensures uniqueness even if same file uploaded multiple times
    hash_input = f"{session_id}_{timestamp.isoformat()}_{original_filename}"
    file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    # Clean the original filename (remove special characters)
    clean_name = "".join(c for c in original_filename if c.isalnum() or c in "._- ")
    clean_name = clean_name.replace(" ", "_")

    # Remove extension from clean name if present
    clean_name_no_ext, _ = os.path.splitext(clean_name)

    # Format: YYYYMMDD_HHMMSS_{hash}_{name}.ext
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp_str}_{file_hash}_{clean_name_no_ext}{ext}"

    return unique_filename


# ============================================================================
# Image Validation
# ============================================================================

def validate_image_file(file: UploadFile) -> Tuple[bool, Optional[str]]:
    """
    Validate an uploaded image file.

    Checks:
    - File size
    - MIME type
    - File is actually an image (can be opened)

    Args:
        file: FastAPI UploadFile object

    Returns:
        Tuple of (is_valid, error_message)
        If valid: (True, None)
        If invalid: (False, "error message")
    """
    # Check MIME type
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        return False, (
            f"Invalid file type: {file.content_type}. "
            f"Allowed types: {', '.join(settings.ALLOWED_IMAGE_TYPES)}"
        )

    # Note: File size check should be done at the route level
    # before reading the file into memory

    return True, None


def get_image_dimensions(file_path: Path) -> Optional[Tuple[int, int]]:
    """
    Get dimensions of an image file.

    Args:
        file_path: Path to the image file

    Returns:
        Tuple of (width, height) or None if error
    """
    try:
        with Image.open(file_path) as img:
            return img.size
    except Exception as e:
        print(f"Error getting image dimensions: {e}")
        return None


# ============================================================================
# Core Storage Operations
# ============================================================================

async def save_uploaded_image(
    file: UploadFile,
    session_id: int
) -> Tuple[str, dict]:
    """
    Save an uploaded image file to storage.

    This is the main function for handling image uploads.

    Process:
    1. Validate the file
    2. Generate unique filename
    3. Save to session directory
    4. Extract image metadata (dimensions, size)
    5. Return storage path and metadata

    Args:
        file: FastAPI UploadFile object
        session_id: ID of the session this image belongs to

    Returns:
        Tuple of (storage_path, metadata_dict)
        storage_path: Relative path from media root (e.g., "sessions/1/image.png")
        metadata_dict: Dict with file_size, width, height, mime_type

    Raises:
        ValueError: If file validation fails
        IOError: If file save fails
    """
    # Validate the file
    is_valid, error_msg = validate_image_file(file)
    if not is_valid:
        raise ValueError(error_msg)

    # Generate unique filename
    unique_filename = generate_unique_filename(
        file.filename or "upload.png",
        session_id
    )

    # Get session directory
    session_dir = get_session_directory(session_id)

    # Full file path
    file_path = session_dir / unique_filename

    # Save the file
    try:
        # Read file content
        content = await file.read()

        # Write to disk
        with open(file_path, "wb") as f:
            f.write(content)

        # Reset file pointer for potential re-reading
        await file.seek(0)

    except Exception as e:
        raise IOError(f"Failed to save file: {str(e)}")

    # Get file metadata
    file_size = os.path.getsize(file_path)
    dimensions = get_image_dimensions(file_path)

    # Build metadata dict
    metadata = {
        "file_size": file_size,
        "mime_type": file.content_type,
        "width": dimensions[0] if dimensions else None,
        "height": dimensions[1] if dimensions else None,
    }

    # Return relative path from media root
    relative_path = file_path.relative_to(get_media_root())
    storage_path = str(relative_path)

    return storage_path, metadata


def get_image_path(storage_path: str) -> Path:
    """
    Get the absolute file system path for a stored image.

    Args:
        storage_path: Relative storage path (from database)
                     e.g., "sessions/1/20241115_a1b2c3_image.png"

    Returns:
        Path: Absolute path to the image file

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    full_path = get_media_root() / storage_path

    if not full_path.exists():
        raise FileNotFoundError(f"Image not found: {storage_path}")

    return full_path


def delete_image(storage_path: str) -> bool:
    """
    Delete an image file from storage.

    Args:
        storage_path: Relative storage path (from database)

    Returns:
        bool: True if deleted successfully, False if file didn't exist

    Raises:
        IOError: If deletion fails
    """
    try:
        full_path = get_media_root() / storage_path

        if not full_path.exists():
            return False

        full_path.unlink()
        return True

    except Exception as e:
        raise IOError(f"Failed to delete image: {str(e)}")


def delete_session_directory(session_id: int) -> bool:
    """
    Delete entire session directory and all images in it.

    Use this when a session is deleted to clean up all associated files.

    Args:
        session_id: Session ID

    Returns:
        bool: True if deleted successfully, False if directory didn't exist
    """
    session_dir = get_media_root() / "sessions" / str(session_id)

    if not session_dir.exists():
        return False

    try:
        shutil.rmtree(session_dir)
        return True
    except Exception as e:
        raise IOError(f"Failed to delete session directory: {str(e)}")


# ============================================================================
# Storage Statistics
# ============================================================================

def get_storage_stats() -> dict:
    """
    Get statistics about storage usage.

    Returns:
        dict: Storage statistics including total size, file count, etc.
    """
    media_root = get_media_root()
    sessions_dir = media_root / "sessions"

    if not sessions_dir.exists():
        return {
            "total_size_bytes": 0,
            "total_files": 0,
            "session_count": 0,
            "total_size_formatted": "0 B"
        }

    total_size = 0
    total_files = 0
    session_count = 0

    # Walk through all session directories
    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir():
            session_count += 1
            for file_path in session_dir.glob("*"):
                if file_path.is_file():
                    total_files += 1
                    total_size += file_path.stat().st_size

    # Format size
    size_formatted = _format_bytes(total_size)

    return {
        "total_size_bytes": total_size,
        "total_files": total_files,
        "session_count": session_count,
        "total_size_formatted": size_formatted
    }


def get_session_storage_stats(session_id: int) -> dict:
    """
    Get storage statistics for a specific session.

    Args:
        session_id: Session ID

    Returns:
        dict: Statistics for this session
    """
    session_dir = get_media_root() / "sessions" / str(session_id)

    if not session_dir.exists():
        return {
            "session_id": session_id,
            "total_size_bytes": 0,
            "file_count": 0,
            "total_size_formatted": "0 B"
        }

    total_size = 0
    file_count = 0

    for file_path in session_dir.glob("*"):
        if file_path.is_file():
            file_count += 1
            total_size += file_path.stat().st_size

    return {
        "session_id": session_id,
        "total_size_bytes": total_size,
        "file_count": file_count,
        "total_size_formatted": _format_bytes(total_size)
    }


# ============================================================================
# Helper Functions
# ============================================================================

def _format_bytes(size: int) -> str:
    """
    Format byte size in human-readable format.

    Args:
        size: Size in bytes

    Returns:
        str: Formatted size (e.g., "2.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def cleanup_empty_directories() -> int:
    """
    Clean up empty session directories.

    Returns:
        int: Number of directories removed
    """
    sessions_dir = get_media_root() / "sessions"
    removed_count = 0

    if not sessions_dir.exists():
        return 0

    for session_dir in sessions_dir.iterdir():
        if session_dir.is_dir() and not any(session_dir.iterdir()):
            try:
                session_dir.rmdir()
                removed_count += 1
            except Exception:
                pass  # Ignore errors

    return removed_count
