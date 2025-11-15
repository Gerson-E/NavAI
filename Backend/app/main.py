"""
Main FastAPI application entry point.

Person A owns this file.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Ultrasound Positioning Platform API",
    description="Backend API for ultrasound operator guidance system",
    version="0.1.0",
)

# CORS middleware - Person A: Configure as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Ultrasound Positioning Platform",
        "version": "0.1.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "database": "not_configured",  # Person A: Update when DB is ready
        "analysis_engine": "stub"  # Will show "ready" when Person B completes implementation
    }


# Person A: Import and include your API routers here
# Example:
# from app.api import sessions, images, reference_views
# app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["sessions"])
# app.include_router(images.router, prefix="/api/v1/images", tags=["images"])
# app.include_router(reference_views.router, prefix="/api/v1/reference-views", tags=["reference-views"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
