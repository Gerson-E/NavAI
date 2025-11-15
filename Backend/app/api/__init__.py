"""
API routers package.

This module exports all API routers for including in the main application.

Usage in main.py:
    from app.api import images, sessions, reference_views
    app.include_router(images.router, prefix="/api/v1", tags=["images"])
"""

from app.api import images, sessions, analysis

__all__ = ["images", "sessions", "analysis"]
