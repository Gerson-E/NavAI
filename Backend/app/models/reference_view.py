"""
Reference View model for storing ideal ultrasound images.

Reference views are the "gold standard" images that user uploads are compared against.
These are curated, high-quality ultrasound images representing correct probe positioning.
"""

from sqlalchemy import Column, String, DateTime, Boolean, Text, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class ReferenceView(Base):
    """
    Reference ultrasound view model.

    Stores metadata about reference images that represent ideal probe positioning.
    User-uploaded images are compared against these references by Person B's
    analysis engine.

    Attributes:
        id: Primary key (string ID like "cardiac_4chamber")
        name: Display name (e.g., "Cardiac 4-Chamber View")
        description: Detailed description of this view
        storage_path: Path where reference image is stored
        thumbnail_path: Path to thumbnail image (for UI display)
        category: Anatomical category (cardiac, abdominal, vascular, etc.)
        is_active: Whether this reference view is currently available
        sort_order: Display order in UI
        created_at: When this reference was added
        updated_at: When this reference was last modified

    Relationships:
        comparisons: All analysis comparisons using this reference view
    """

    __tablename__ = "reference_views"

    # ========================================================================
    # Columns
    # ========================================================================

    id = Column(
        String(50),
        primary_key=True,
        doc="Unique reference view identifier (e.g., 'cardiac_4chamber')"
    )

    name = Column(
        String(255),
        nullable=False,
        doc="Display name for this reference view"
    )

    description = Column(
        Text,
        nullable=True,
        doc="Detailed description of this ultrasound view"
    )

    storage_path = Column(
        String(512),
        nullable=False,
        unique=True,
        doc="Path where reference image is stored"
    )

    thumbnail_path = Column(
        String(512),
        nullable=True,
        doc="Path to thumbnail image (for UI display)"
    )

    category = Column(
        String(50),
        nullable=False,
        index=True,
        doc="Anatomical category (cardiac, abdominal, vascular, etc.)"
    )

    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
        doc="Whether this reference view is currently available"
    )

    sort_order = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Display order in UI (lower numbers first)"
    )

    # ========================================================================
    # Timestamps
    # ========================================================================

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="When this reference view was added"
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="When this reference view was last updated"
    )

    # ========================================================================
    # Relationships
    # ========================================================================

    comparisons = relationship(
        "Comparison",
        back_populates="reference_view",
        cascade="all, delete-orphan",
        doc="All analysis comparisons using this reference view"
    )

    # ========================================================================
    # Methods
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<ReferenceView(id='{self.id}', name='{self.name}', "
            f"category='{self.category}', active={self.is_active})>"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.name} ({self.category})"

    @property
    def comparison_count(self) -> int:
        """Get the number of comparisons using this reference view."""
        return len(self.comparisons)

    def activate(self) -> None:
        """Activate this reference view."""
        self.is_active = True

    def deactivate(self) -> None:
        """Deactivate this reference view (soft delete)."""
        self.is_active = False


# ============================================================================
# Predefined Reference View Categories
# ============================================================================

class ReferenceViewCategory:
    """
    Predefined categories for reference views.

    These categories help organize reference views in the UI and
    ensure consistency across the application.
    """

    CARDIAC = "cardiac"
    ABDOMINAL = "abdominal"
    VASCULAR = "vascular"
    OBSTETRIC = "obstetric"
    MUSCULOSKELETAL = "musculoskeletal"
    OTHER = "other"

    @classmethod
    def all(cls) -> list[str]:
        """Get all available categories."""
        return [
            cls.CARDIAC,
            cls.ABDOMINAL,
            cls.VASCULAR,
            cls.OBSTETRIC,
            cls.MUSCULOSKELETAL,
            cls.OTHER,
        ]

    @classmethod
    def is_valid(cls, category: str) -> bool:
        """Check if a category is valid."""
        return category in cls.all()
