"""
Database configuration and session management.

This module sets up SQLAlchemy engine, session factory, and base class.
It provides the database session dependency for FastAPI routes.
"""

from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings


# ============================================================================
# SQLAlchemy Engine Setup
# ============================================================================

# Create database engine
# - echo: Log all SQL statements (useful for debugging)
# - connect_args: SQLite-specific settings (not needed for PostgreSQL)
connect_args = {}
if settings.database_is_sqlite:
    # SQLite-specific: Enable foreign key constraints and set timeout
    connect_args = {
        "check_same_thread": False,  # Allow multiple threads (needed for FastAPI)
        "timeout": 30  # Connection timeout in seconds
    }

engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,  # Set to True to see SQL queries
    connect_args=connect_args,
    pool_pre_ping=True,  # Verify connections before using them
)


# ============================================================================
# Enable Foreign Key Support for SQLite
# ============================================================================

if settings.database_is_sqlite:
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        """
        Enable foreign key constraints in SQLite.
        SQLite has foreign keys disabled by default - this enables them.
        """
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


# ============================================================================
# Session Factory
# ============================================================================

# SessionLocal is a factory for creating database sessions
# Each session represents a "unit of work" with the database
SessionLocal = sessionmaker(
    autocommit=False,  # Require explicit commit
    autoflush=False,   # Don't auto-flush before queries
    bind=engine        # Use the engine we created above
)


# ============================================================================
# Declarative Base
# ============================================================================

# All database models inherit from this Base class
# SQLAlchemy uses this to track all models and generate tables
Base = declarative_base()


# ============================================================================
# Database Session Dependency
# ============================================================================

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session.

    Usage in routes:
        @router.get("/items")
        def get_items(db: Session = Depends(get_db)):
            items = db.query(Item).all()
            return items

    How it works:
        1. Creates a new session when route is called
        2. Yields the session to the route function
        3. Automatically closes the session after route completes
        4. Rolls back on exception to maintain data integrity

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        # If an error occurs, roll back any changes
        db.rollback()
        raise
    finally:
        # Always close the session, even if an error occurred
        db.close()


# ============================================================================
# Database Initialization Utilities
# ============================================================================

def create_tables() -> None:
    """
    Create all database tables.

    This creates tables for all models that inherit from Base.
    Use this for development. For production, use Alembic migrations.

    Example:
        from app.core.database import create_tables
        create_tables()
    """
    Base.metadata.create_all(bind=engine)


def drop_tables() -> None:
    """
    Drop all database tables.

    WARNING: This deletes all data! Only use in development/testing.

    Example:
        from app.core.database import drop_tables
        drop_tables()
    """
    Base.metadata.drop_all(bind=engine)


def reset_database() -> None:
    """
    Drop and recreate all tables.

    WARNING: This deletes all data! Only use in development/testing.

    Example:
        from app.core.database import reset_database
        reset_database()
    """
    drop_tables()
    create_tables()


# ============================================================================
# Database Health Check
# ============================================================================

def check_database_connection() -> bool:
    """
    Check if database connection is working.

    Returns:
        bool: True if connection is successful, False otherwise

    Example:
        from app.core.database import check_database_connection
        if check_database_connection():
            print("Database connected!")
    """
    try:
        # Try to connect and execute a simple query
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False
