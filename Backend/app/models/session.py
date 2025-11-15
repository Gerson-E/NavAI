"""
Session model for grouping ultrasound scans.

A session represents one ultrasound scanning session (typically one patient visit).
It groups together all images captured during that session and tracks session metadata.
"""

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class SessionStatus(str, enum.Enum):
    """
    Possible states for a scan session.

    ACTIVE: Session is currently in progress
    COMPLETED: Session has been finished successfully
    CANCELLED: Session was cancelled or abandoned
    """
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Session(Base):
    """
    Scan session model.

    Represents a single ultrasound scanning session. Groups together all images
    captured during one patient encounter.

    Attributes:
        id: Primary key
        user_id: Foreign key to user who created this session
        patient_identifier: Optional patient reference (NOT PHI - use study ID, not name)
        status: Current session status (active, completed, cancelled)
        notes: Optional session notes or comments
        created_at: When session was started
        updated_at: When session was last modified

    Relationships:
        user: The user who created this session
        images: All images captured during this session
        comparisons: All analysis comparisons performed in this session
    """

    __tablename__ = "sessions"

    # ========================================================================
    # Columns
    # ========================================================================

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        doc="Unique session identifier"
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of user who created this session"
    )

    patient_identifier = Column(
        String(100),
        nullable=True,
        index=True,
        doc="Optional patient/study identifier (NOT PHI - use anonymized ID)"
    )

    status = Column(
        Enum(SessionStatus),
        default=SessionStatus.ACTIVE,
        nullable=False,
        index=True,
        doc="Current session status"
    )

    notes = Column(
        Text,
        nullable=True,
        doc="Optional notes or comments about this session"
    )

    # ========================================================================
    # Timestamps
    # ========================================================================

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        doc="When the session was started"
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="When the session was last updated"
    )

    # ========================================================================
    # Relationships
    # ========================================================================

    user = relationship(
        "User",
        back_populates="sessions",
        doc="The user who created this session"
    )

    images = relationship(
        "Image",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Image.created_at",
        doc="All images captured during this session"
    )

    comparisons = relationship(
        "Comparison",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Comparison.created_at.desc()",
        doc="All analysis comparisons performed in this session"
    )

    # ========================================================================
    # Methods
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<Session(id={self.id}, user_id={self.user_id}, "
            f"status={self.status.value}, images={len(self.images)})>"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        patient_info = f" - {self.patient_identifier}" if self.patient_identifier else ""
        return f"Session {self.id}{patient_info} ({self.status.value})"

    @property
    def image_count(self) -> int:
        """Get the number of images in this session."""
        return len(self.images)

    @property
    def comparison_count(self) -> int:
        """Get the number of comparisons in this session."""
        return len(self.comparisons)

    def is_active(self) -> bool:
        """Check if the session is currently active."""
        return self.status == SessionStatus.ACTIVE

    def mark_completed(self) -> None:
        """Mark the session as completed."""
        self.status = SessionStatus.COMPLETED

    def mark_cancelled(self) -> None:
        """Mark the session as cancelled."""
        self.status = SessionStatus.CANCELLED
