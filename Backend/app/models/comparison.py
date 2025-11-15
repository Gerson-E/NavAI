"""
Comparison model for storing image analysis results.

This model stores the results from Person B's image analysis engine.
Each comparison represents one analysis of a user image against a reference view.
"""

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, Enum, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class ComparisonVerdict(str, enum.Enum):
    """
    Possible verdicts from image analysis.

    GOOD: Probe positioning is correct
    BORDERLINE: Positioning needs minor adjustment
    POOR: Positioning is incorrect, needs significant adjustment
    """
    GOOD = "good"
    BORDERLINE = "borderline"
    POOR = "poor"


class Comparison(Base):
    """
    Image comparison/analysis result model.

    Stores results from Person B's analysis engine when comparing a user-uploaded
    ultrasound image against a reference view. This is the core data model that
    bridges your API layer with Person B's CV analysis.

    This model maps directly to the ComparisonResult TypedDict from
    app.analysis.interface.py, with additional metadata for database storage.

    Attributes:
        id: Primary key
        session_id: Foreign key to session this comparison belongs to
        image_id: Foreign key to the image that was analyzed
        reference_view_id: Foreign key to the reference view used for comparison

        # Analysis results (from Person B's engine)
        ssim: Structural Similarity Index (0.0 to 1.0)
        ncc: Normalized Cross-Correlation (-1.0 to 1.0)
        verdict: Overall assessment (good, borderline, poor)
        message: Human-readable feedback for the operator
        confidence: Confidence score of the analysis (0.0 to 1.0)

        # Metadata
        processing_time_ms: How long the analysis took (milliseconds)
        created_at: When the comparison was performed

    Relationships:
        session: The session this comparison belongs to
        image: The image that was analyzed
        reference_view: The reference view used for comparison
    """

    __tablename__ = "comparisons"

    # ========================================================================
    # Columns - Primary Key & Foreign Keys
    # ========================================================================

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        doc="Unique comparison identifier"
    )

    session_id = Column(
        Integer,
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of session this comparison belongs to"
    )

    image_id = Column(
        Integer,
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of image that was analyzed"
    )

    reference_view_id = Column(
        String(50),
        ForeignKey("reference_views.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        doc="ID of reference view used for comparison"
    )

    # ========================================================================
    # Analysis Results (from Person B's engine)
    # ========================================================================

    ssim = Column(
        Float,
        nullable=False,
        doc="Structural Similarity Index (0.0 to 1.0, higher is better)"
    )

    ncc = Column(
        Float,
        nullable=False,
        doc="Normalized Cross-Correlation (-1.0 to 1.0, higher is better)"
    )

    verdict = Column(
        Enum(ComparisonVerdict),
        nullable=False,
        index=True,
        doc="Overall assessment: good, borderline, or poor"
    )

    message = Column(
        Text,
        nullable=False,
        doc="Human-readable feedback message for the operator"
    )

    confidence = Column(
        Float,
        nullable=False,
        doc="Confidence score of the analysis (0.0 to 1.0)"
    )

    # ========================================================================
    # Metadata
    # ========================================================================

    processing_time_ms = Column(
        Integer,
        nullable=True,
        doc="How long the analysis took in milliseconds"
    )

    # ========================================================================
    # Timestamps
    # ========================================================================

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        doc="When the comparison was performed"
    )

    # ========================================================================
    # Relationships
    # ========================================================================

    session = relationship(
        "Session",
        back_populates="comparisons",
        doc="The session this comparison belongs to"
    )

    image = relationship(
        "Image",
        back_populates="comparisons",
        doc="The image that was analyzed"
    )

    reference_view = relationship(
        "ReferenceView",
        back_populates="comparisons",
        doc="The reference view used for comparison"
    )

    # ========================================================================
    # Methods
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<Comparison(id={self.id}, image_id={self.image_id}, "
            f"ref='{self.reference_view_id}', verdict={self.verdict.value}, "
            f"ssim={self.ssim:.2f})>"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Comparison #{self.id}: {self.verdict.value} "
            f"(SSIM: {self.ssim:.2f}, NCC: {self.ncc:.2f})"
        )

    @property
    def is_good(self) -> bool:
        """Check if verdict is 'good'."""
        return self.verdict == ComparisonVerdict.GOOD

    @property
    def is_borderline(self) -> bool:
        """Check if verdict is 'borderline'."""
        return self.verdict == ComparisonVerdict.BORDERLINE

    @property
    def is_poor(self) -> bool:
        """Check if verdict is 'poor'."""
        return self.verdict == ComparisonVerdict.POOR

    @property
    def processing_time_seconds(self) -> float:
        """Get processing time in seconds."""
        if self.processing_time_ms is not None:
            return self.processing_time_ms / 1000.0
        return 0.0

    def to_dict(self) -> dict:
        """
        Convert comparison to dictionary format.

        This matches the ComparisonResult TypedDict from Person B's interface.

        Returns:
            dict: Dictionary with analysis results
        """
        return {
            "ssim": self.ssim,
            "ncc": self.ncc,
            "verdict": self.verdict.value,
            "message": self.message,
            "confidence": self.confidence,
        }

    @classmethod
    def from_analysis_result(
        cls,
        session_id: int,
        image_id: int,
        reference_view_id: str,
        result: dict,
        processing_time_ms: int = None
    ) -> "Comparison":
        """
        Create a Comparison from Person B's analysis result.

        This is a factory method that converts the ComparisonResult from
        Person B's engine into a database model.

        Args:
            session_id: ID of the session
            image_id: ID of the analyzed image
            reference_view_id: ID of the reference view
            result: ComparisonResult dict from Person B's engine
            processing_time_ms: Optional processing time in milliseconds

        Returns:
            Comparison: New Comparison instance (not yet saved to DB)

        Example:
            from app.analysis.engine import compare_to_reference

            result = compare_to_reference(image_path, "cardiac_4chamber")
            comparison = Comparison.from_analysis_result(
                session_id=1,
                image_id=42,
                reference_view_id="cardiac_4chamber",
                result=result,
                processing_time_ms=250
            )
            db.add(comparison)
            db.commit()
        """
        return cls(
            session_id=session_id,
            image_id=image_id,
            reference_view_id=reference_view_id,
            ssim=result["ssim"],
            ncc=result["ncc"],
            verdict=ComparisonVerdict(result["verdict"]),
            message=result["message"],
            confidence=result["confidence"],
            processing_time_ms=processing_time_ms,
        )
