"""
User model for authentication and session ownership.

This model stores user accounts for the ultrasound positioning platform.
Even though authentication isn't implemented yet, we create this now so
sessions can have proper foreign key relationships from the start.
"""

from sqlalchemy import Boolean, Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class User(Base):
    """
    User account model.

    Attributes:
        id: Primary key
        email: Unique email address (used for login)
        hashed_password: Bcrypt hashed password (NEVER store plain passwords)
        full_name: User's full name (optional)
        is_active: Account active status (for soft delete / suspension)
        is_superuser: Admin/superuser flag (for future admin features)
        created_at: Account creation timestamp
        updated_at: Last modification timestamp

    Relationships:
        sessions: All scan sessions created by this user
    """

    __tablename__ = "users"

    # ========================================================================
    # Columns
    # ========================================================================

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        doc="Unique user identifier"
    )

    email = Column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
        doc="User's email address (used for login)"
    )

    hashed_password = Column(
        String(255),
        nullable=False,
        doc="Bcrypt hashed password (never store plain text)"
    )

    full_name = Column(
        String(255),
        nullable=True,
        doc="User's full name"
    )

    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Whether the account is active (False = suspended)"
    )

    is_superuser = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Whether the user has admin privileges"
    )

    # ========================================================================
    # Timestamps
    # ========================================================================

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="When the user account was created"
    )

    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="When the user account was last updated"
    )

    # ========================================================================
    # Relationships
    # ========================================================================

    sessions = relationship(
        "Session",
        back_populates="user",
        cascade="all, delete-orphan",
        doc="All scan sessions created by this user"
    )

    # ========================================================================
    # Methods
    # ========================================================================

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"<User(id={self.id}, email='{self.email}', active={self.is_active})>"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.full_name or self.email}"
