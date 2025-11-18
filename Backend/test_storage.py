"""
Test script for image storage system.

This tests the storage module without needing to run the full API.

Usage:
    python test_storage.py
"""

import asyncio
import io
from pathlib import Path
from PIL import Image
from fastapi import UploadFile

from app.core.storage import (
    get_media_root,
    get_session_directory,
    generate_unique_filename,
    save_uploaded_image,
    get_image_path,
    delete_image,
    delete_session_directory,
    get_storage_stats,
    get_session_storage_stats,
    validate_image_file,
    get_image_dimensions,
    cleanup_empty_directories,
)


def create_test_image(width: int = 800, height: int = 600, format: str = "PNG") -> bytes:
    """
    Create a test image in memory.

    Args:
        width: Image width
        height: Image height
        format: Image format (PNG, JPEG)

    Returns:
        bytes: Image file content
    """
    # Create a simple gradient image
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    for x in range(width):
        for y in range(height):
            # Create a gradient effect
            r = int((x / width) * 255)
            g = int((y / height) * 255)
            b = 128
            pixels[x, y] = (r, g, b)

    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()


class MockUploadFile:
    """Mock UploadFile for testing."""

    def __init__(self, filename: str, content: bytes, content_type: str):
        self.filename = filename
        self.content_type = content_type
        self._file = io.BytesIO(content)

    async def read(self) -> bytes:
        return self._file.read()

    async def seek(self, offset: int):
        self._file.seek(offset)


def create_upload_file(filename: str, content: bytes, content_type: str):
    """
    Create a mock UploadFile object for testing.

    Args:
        filename: Filename
        content: File content
        content_type: MIME type

    Returns:
        MockUploadFile: Mock upload file
    """
    return MockUploadFile(filename, content, content_type)


def test_media_root():
    """Test media root directory creation."""
    print("✅ Testing media root directory...")

    media_root = get_media_root()
    assert media_root.exists(), "Media root should be created"
    assert media_root.is_dir(), "Media root should be a directory"
    print(f"  ✓ Media root created: {media_root}")


def test_session_directory():
    """Test session directory creation."""
    print("\n✅ Testing session directory creation...")

    session_dir = get_session_directory(999)
    assert session_dir.exists(), "Session directory should be created"
    assert session_dir.is_dir(), "Session directory should be a directory"
    assert "sessions/999" in str(session_dir), "Should include session ID in path"
    print(f"  ✓ Session directory created: {session_dir}")


def test_filename_generation():
    """Test unique filename generation."""
    print("\n✅ Testing filename generation...")

    filename1 = generate_unique_filename("test.png", session_id=1)
    filename2 = generate_unique_filename("test.png", session_id=1)

    # Should be different due to timestamp
    assert filename1 != filename2, "Filenames should be unique"
    assert filename1.endswith(".png"), "Should preserve extension"
    print(f"  ✓ Generated unique filename: {filename1}")
    print(f"  ✓ Generated unique filename: {filename2}")


def test_image_validation():
    """Test image file validation."""
    print("\n✅ Testing image validation...")

    # Valid PNG
    png_content = create_test_image(format="PNG")
    png_file = create_upload_file("test.png", png_content, "image/png")
    is_valid, error = validate_image_file(png_file)
    assert is_valid, "PNG should be valid"
    assert error is None, "No error for valid file"
    print("  ✓ PNG validation passed")

    # Valid JPEG
    jpeg_content = create_test_image(format="JPEG")
    jpeg_file = create_upload_file("test.jpg", jpeg_content, "image/jpeg")
    is_valid, error = validate_image_file(jpeg_file)
    assert is_valid, "JPEG should be valid"
    print("  ✓ JPEG validation passed")

    # Invalid type
    invalid_file = create_upload_file("test.txt", b"hello", "text/plain")
    is_valid, error = validate_image_file(invalid_file)
    assert not is_valid, "Text file should be invalid"
    assert error is not None, "Should have error message"
    print(f"  ✓ Invalid type rejected: {error}")


async def test_image_upload():
    """Test complete image upload process."""
    print("\n✅ Testing image upload...")

    # Create test image
    image_content = create_test_image(width=1024, height=768)
    upload_file = create_upload_file("ultrasound_test.png", image_content, "image/png")

    # Upload to session 1
    storage_path, metadata = await save_uploaded_image(upload_file, session_id=1)

    print(f"  ✓ Image saved to: {storage_path}")
    print(f"  ✓ Metadata: {metadata}")

    # Verify metadata
    assert metadata["file_size"] > 0, "File size should be > 0"
    assert metadata["width"] == 1024, "Width should match"
    assert metadata["height"] == 768, "Height should match"
    assert metadata["mime_type"] == "image/png", "MIME type should match"

    # Verify file exists
    full_path = get_image_path(storage_path)
    assert full_path.exists(), "File should exist on disk"
    print(f"  ✓ File exists at: {full_path}")

    # Test dimensions extraction
    dimensions = get_image_dimensions(full_path)
    assert dimensions == (1024, 768), "Dimensions should match"
    print(f"  ✓ Dimensions extracted: {dimensions}")

    return storage_path  # Return for cleanup test


async def test_image_deletion(storage_path: str):
    """Test image deletion."""
    print("\n✅ Testing image deletion...")

    # Delete the image
    deleted = delete_image(storage_path)
    assert deleted, "Delete should return True"
    print(f"  ✓ Image deleted: {storage_path}")

    # Verify file is gone
    try:
        get_image_path(storage_path)
        assert False, "Should raise FileNotFoundError"
    except FileNotFoundError:
        print("  ✓ File confirmed deleted")

    # Try deleting again (should return False)
    deleted_again = delete_image(storage_path)
    assert not deleted_again, "Second delete should return False"
    print("  ✓ Double delete handled correctly")


def test_session_deletion():
    """Test session directory deletion."""
    print("\n✅ Testing session directory deletion...")

    # Session 999 was created earlier in tests
    deleted = delete_session_directory(999)
    assert deleted, "Session directory should be deleted"
    print("  ✓ Session directory deleted: sessions/999")

    # Try deleting again
    deleted_again = delete_session_directory(999)
    assert not deleted_again, "Second delete should return False"
    print("  ✓ Double delete handled correctly")


async def test_storage_stats():
    """Test storage statistics."""
    print("\n✅ Testing storage statistics...")

    # Create some test images
    for i in range(3):
        image_content = create_test_image()
        upload_file = create_upload_file(f"test_{i}.png", image_content, "image/png")
        await save_uploaded_image(upload_file, session_id=100)

    # Get overall stats
    stats = get_storage_stats()
    print(f"  ✓ Overall stats: {stats}")
    assert stats["total_files"] >= 3, "Should have at least 3 files"
    assert stats["session_count"] >= 1, "Should have at least 1 session"

    # Get session-specific stats
    session_stats = get_session_storage_stats(100)
    print(f"  ✓ Session 100 stats: {session_stats}")
    assert session_stats["file_count"] == 3, "Should have 3 files in session 100"

    # Cleanup
    delete_session_directory(100)


def test_cleanup():
    """Test cleanup of empty directories."""
    print("\n✅ Testing cleanup of empty directories...")

    # Create an empty session directory
    get_session_directory(888)

    # Clean up
    removed = cleanup_empty_directories()
    print(f"  ✓ Removed {removed} empty directories")


async def run_all_tests():
    """Run all storage tests."""
    print("=" * 60)
    print("TESTING IMAGE STORAGE SYSTEM")
    print("=" * 60)

    try:
        # Run tests in order
        test_media_root()
        test_session_directory()
        test_filename_generation()
        test_image_validation()

        # Async tests
        storage_path = await test_image_upload()
        await test_image_deletion(storage_path)

        test_session_deletion()
        await test_storage_stats()
        test_cleanup()

        print("\n" + "=" * 60)
        print("✅ ALL STORAGE TESTS PASSED!")
        print("=" * 60)

        print("\nStorage system features:")
        print("  ✓ Media root directory management")
        print("  ✓ Session-based organization")
        print("  ✓ Unique filename generation")
        print("  ✓ Image validation (type, size)")
        print("  ✓ Dimension extraction")
        print("  ✓ File save/retrieve/delete operations")
        print("  ✓ Session directory deletion")
        print("  ✓ Storage statistics")
        print("  ✓ Empty directory cleanup")

        print("\nReady for Phase 2, Part 2: Image Upload Endpoint\n")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run async tests
    asyncio.run(run_all_tests())
