"""
Database models package.

This module exports all database models for easy importing.
All models inherit from the Base class defined in app.core.database.

Usage:
    from app.models import User, Session, Image, ReferenceView, Comparison

Import this in alembic/env.py for migrations:
    from app.models import Base  # This imports all models
"""

from app.models.user import User
from app.models.session import Session, SessionStatus
from app.models.image import Image
from app.models.reference_view import ReferenceView, ReferenceViewCategory
from app.models.comparison import Comparison, ComparisonVerdict
from app.models.classification import Classification

# Export all models
__all__ = [
    # Models
    "User",
    "Session",
    "Image",
    "ReferenceView",
    "Comparison",
    "Classification",

    # Enums
    "SessionStatus",
    "ComparisonVerdict",
    "ReferenceViewCategory",
]
