"""
Image model for storing ultrasound image metadata.

This model stores metadata about uploaded ultrasound images.
The actual image files are stored on disk, and this model tracks their location
and properties.
"""

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean, BigInteger
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Image(Base):
    """
    Ultrasound image metadata model.

    Stores metadata about uploaded images. The actual image files are stored
    on the filesystem (not in the database), and this model tracks their location.

    Attributes:
        id: Primary key
        session_id: Foreign key to the session this image belongs to
        filename: Original filename from upload
        storage_path: Path where file is stored on disk (relative to MEDIA_ROOT)
        file_size: File size in bytes
        mime_type: MIME type (image/png, image/jpeg, etc.)
        width: Image width in pixels
        height: Image height in pixels
        is_reference: Whether this is a reference view (False for user uploads)
        created_at: When the image was uploaded

    Relationships:
        session: The session this image belongs to
        comparisons: All analysis comparisons using this image
    """

    __tablename__ = "images"

    # ========================================================================
    # Columns
    # ========================================================================

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        doc="Unique image identifier"
    )

    session_id = Column(
        Integer,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of session this image belongs to"
    )

    filename = Column(
        String(255),
        nullable=False,
        doc="Original filename from upload"
    )

    storage_path = Column(
        String(512),
        nullable=False,
        unique=True,
        doc="Path where file is stored (relative to MEDIA_ROOT)"
    )

    file_size = Column(
        BigInteger,
        nullable=False,
        doc="File size in bytes"
    )

    mime_type = Column(
        String(50),
        nullable=False,
        doc="MIME type (image/png, image/jpeg, etc.)"
    )

    width = Column(
        Integer,
        nullable=True,
        doc="Image width in pixels"
    )

    height = Column(
        Integer,
        nullable=True,
        doc="Image height in pixels"
    )

    is_reference = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Whether this is a reference image (False for user uploads)"
    )

    # ========================================================================
    # Timestamps
    # ========================================================================

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        doc="When the image was uploaded"
    )

    # ========================================================================
    # Relationships
    # ========================================================================

    session = relationship(
        "Session",
        back_populates="images",
        doc="The session this image belongs to"
    )

    comparisons = relationship(
        "Comparison",
        back_populates="image",
        cascade="all, delete-orphan",
        doc="All analysis comparisons using this image"
    )

    # ========================================================================
    # Methods
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<Image(id={self.id}, filename='{self.filename}', "
            f"session_id={self.session_id}, size={self.file_size})>"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.filename} ({self.format_file_size()})"

    @property
    def dimensions(self) -> str:
        """Get image dimensions as a string."""
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "unknown"

    def format_file_size(self) -> str:
        """
        Format file size in human-readable format.

        Returns:
            str: Formatted size (e.g., "2.5 MB", "150 KB")
        """
        size = self.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    @property
    def is_png(self) -> bool:
        """Check if image is PNG format."""
        return self.mime_type == "image/png"

    @property
    def is_jpeg(self) -> bool:
        """Check if image is JPEG format."""
        return self.mime_type in ("image/jpeg", "image/jpg")

    def get_file_extension(self) -> str:
        """
        Get file extension from MIME type.

        Returns:
            str: File extension (e.g., "png", "jpg")
        """
        mime_to_ext = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
        }
        return mime_to_ext.get(self.mime_type, "bin")
