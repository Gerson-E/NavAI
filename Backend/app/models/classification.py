"""
Classification model for storing organ detection results.

This model stores the results from Person B's organ classification engine.
Each classification represents one analysis of determining what organ appears
in an ultrasound image. This is the MVP feature for kidney detection.
"""

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Classification(Base):
    """
    Organ classification result model.

    Stores results from Person B's classification engine when analyzing
    what organ is shown in an ultrasound image. This is the MVP feature
    for kidney detection.

    This model maps directly to the ClassificationResult TypedDict from
    app.analysis.interface.py, with additional metadata for database storage.

    Attributes:
        id: Primary key
        session_id: Foreign key to session this classification belongs to
        image_id: Foreign key to the image that was classified

        # Classification results (from Person B's engine)
        detected_organ: The identified organ or anatomy
        confidence: Confidence score of the classification (0.0 to 1.0)
        is_kidney: Boolean flag for kidney detection (MVP feature)
        message: Human-readable feedback for the operator

        # Metadata
        processing_time_ms: How long the classification took (milliseconds)
        created_at: When the classification was performed

    Relationships:
        session: The session this classification belongs to
        image: The image that was classified
    """

    __tablename__ = "classifications"

    # ========================================================================
    # Columns - Primary Key & Foreign Keys
    # ========================================================================

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        doc="Unique classification identifier"
    )

    session_id = Column(
        Integer,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of session this classification belongs to"
    )

    image_id = Column(
        Integer,
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of image that was classified"
    )

    # ========================================================================
    # Classification Results (from Person B's engine)
    # ========================================================================

    detected_organ = Column(
        String(100),
        nullable=False,
        index=True,
        doc="The identified organ or anatomy (e.g., 'kidney', 'liver', 'heart', 'unknown')"
    )

    confidence = Column(
        Float,
        nullable=False,
        doc="Confidence score of the classification (0.0 to 1.0)"
    )

    is_kidney = Column(
        Boolean,
        nullable=False,
        index=True,
        doc="Boolean flag indicating if the image shows a kidney (MVP feature)"
    )

    message = Column(
        Text,
        nullable=False,
        doc="Human-readable feedback message for the operator"
    )

    # ========================================================================
    # Metadata
    # ========================================================================

    processing_time_ms = Column(
        Integer,
        nullable=True,
        doc="How long the classification took in milliseconds"
    )

    # ========================================================================
    # Timestamps
    # ========================================================================

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        doc="When the classification was performed"
    )

    # ========================================================================
    # Relationships
    # ========================================================================

    session = relationship(
        "Session",
        back_populates="classifications",
        doc="The session this classification belongs to"
    )

    image = relationship(
        "Image",
        back_populates="classifications",
        doc="The image that was classified"
    )

    # ========================================================================
    # Methods
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<Classification(id={self.id}, image_id={self.image_id}, "
            f"organ='{self.detected_organ}', is_kidney={self.is_kidney}, "
            f"confidence={self.confidence:.2f})>"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        kidney_status = "Kidney detected" if self.is_kidney else "Not a kidney"
        return (
            f"Classification #{self.id}: {self.detected_organ} "
            f"({kidney_status}, confidence: {self.confidence:.2f})"
        )

    @property
    def processing_time_seconds(self) -> float:
        """Get processing time in seconds."""
        if self.processing_time_ms is not None:
            return self.processing_time_ms / 1000.0
        return 0.0

    @property
    def is_high_confidence(self) -> bool:
        """Check if confidence is high (>= 0.8)."""
        return self.confidence >= 0.8

    @property
    def is_low_confidence(self) -> bool:
        """Check if confidence is low (< 0.5)."""
        return self.confidence < 0.5

    def to_dict(self) -> dict:
        """
        Convert classification to dictionary format.

        This matches the ClassificationResult TypedDict from Person B's interface.

        Returns:
            dict: Dictionary with classification results
        """
        return {
            "detected_organ": self.detected_organ,
            "confidence": self.confidence,
            "is_kidney": self.is_kidney,
            "message": self.message,
        }

    @classmethod
    def from_classification_result(
        cls,
        session_id: int,
        image_id: int,
        result: dict,
        processing_time_ms: int = None
    ) -> "Classification":
        """
        Create a Classification from Person B's classification result.

        This is a factory method that converts the ClassificationResult from
        Person B's engine into a database model.

        Args:
            session_id: ID of the session
            image_id: ID of the classified image
            result: ClassificationResult dict from Person B's engine
            processing_time_ms: Optional processing time in milliseconds

        Returns:
            Classification: New Classification instance (not yet saved to DB)

        Example:
            from app.analysis.engine import classify_organ

            result = classify_organ(image_path)
            classification = Classification.from_classification_result(
                session_id=1,
                image_id=42,
                result=result,
                processing_time_ms=180
            )
            db.add(classification)
            db.commit()
        """
        return cls(
            session_id=session_id,
            image_id=image_id,
            detected_organ=result["detected_organ"],
            confidence=result["confidence"],
            is_kidney=result["is_kidney"],
            message=result["message"],
            processing_time_ms=processing_time_ms,
        )
