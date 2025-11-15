"""
Application configuration using Pydantic Settings.

This module manages all configuration from environment variables with type validation.
Settings are loaded from .env file and can be overridden by environment variables.
"""

from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import secrets


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via .env file or environment variables.
    Pydantic validates types and provides defaults.
    """

    # ============================================================================
    # General Application Settings
    # ============================================================================

    PROJECT_NAME: str = "Ultrasound Positioning Platform API"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "Backend API for ultrasound operator guidance system"

    ENVIRONMENT: str = Field(
        default="development",
        description="Environment: development, staging, or production"
    )

    DEBUG: bool = Field(
        default=True,
        description="Enable debug mode (disable in production)"
    )

    # ============================================================================
    # API Configuration
    # ============================================================================

    API_V1_PREFIX: str = "/api/v1"

    # CORS settings
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins (frontend URLs)"
    )

    # ============================================================================
    # Database Configuration
    # ============================================================================

    DATABASE_URL: str = Field(
        default="sqlite:///./navai.db",
        description="Database connection URL. Examples:\n"
                    "  - SQLite: sqlite:///./navai.db\n"
                    "  - PostgreSQL: postgresql://user:pass@localhost/dbname\n"
                    "  - MySQL: mysql://user:pass@localhost/dbname"
    )

    DB_ECHO: bool = Field(
        default=False,
        description="Echo SQL queries to console (useful for debugging)"
    )

    # ============================================================================
    # File Storage Configuration
    # ============================================================================

    MEDIA_ROOT: str = Field(
        default="./media",
        description="Root directory for uploaded files"
    )

    REFERENCE_VIEWS_PATH: str = Field(
        default="./reference_views",
        description="Directory containing reference ultrasound images"
    )

    MAX_UPLOAD_SIZE: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        description="Maximum file upload size in bytes"
    )

    ALLOWED_IMAGE_TYPES: List[str] = Field(
        default=["image/png", "image/jpeg", "image/jpg"],
        description="Allowed MIME types for image uploads"
    )

    # ============================================================================
    # Security Configuration
    # ============================================================================

    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key for JWT encoding (CHANGE IN PRODUCTION)"
    )

    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60 * 24 * 7,  # 7 days
        description="JWT access token expiration time in minutes"
    )

    # Password hashing
    PASSWORD_HASH_ALGORITHM: str = "bcrypt"

    # ============================================================================
    # Analysis Engine Configuration
    # ============================================================================

    ANALYSIS_TIMEOUT_SECONDS: int = Field(
        default=30,
        description="Maximum time allowed for image analysis"
    )

    # Default reference view IDs (should match Person B's implementation)
    DEFAULT_REFERENCE_VIEWS: List[str] = Field(
        default=[
            "cardiac_4chamber",
            "cardiac_parasternal_long",
            "liver_standard",
            "kidney_longitudinal"
        ],
        description="Default reference view identifiers"
    )

    # ============================================================================
    # Logging Configuration
    # ============================================================================

    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )

    # ============================================================================
    # Pagination Defaults
    # ============================================================================

    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100

    # ============================================================================
    # Validators
    # ============================================================================

    @field_validator('ENVIRONMENT')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ensure environment is one of the allowed values."""
        allowed = ['development', 'staging', 'production']
        if v.lower() not in allowed:
            raise ValueError(f'ENVIRONMENT must be one of {allowed}')
        return v.lower()

    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in allowed:
            raise ValueError(f'LOG_LEVEL must be one of {allowed}')
        return v.upper()

    # ============================================================================
    # Computed Properties
    # ============================================================================

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT == "development"

    @property
    def database_is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return self.DATABASE_URL.startswith("sqlite")

    # ============================================================================
    # Model Configuration
    # ============================================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore extra fields in .env
    )


# Global settings instance
# Import this in other modules: from app.core.config import settings
settings = Settings()
