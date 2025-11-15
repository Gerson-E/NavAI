"""
Session API endpoints.

This module provides endpoints for managing ultrasound scanning sessions.

Endpoints:
    POST   /sessions              - Create new session
    GET    /sessions              - List sessions (paginated)
    GET    /sessions/{id}         - Get session details
    PATCH  /sessions/{id}         - Update session
    DELETE /sessions/{id}         - Delete session
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session as DBSession

from app.core.database import get_db
from app.core.storage import delete_session_directory
from app.models import Session, SessionStatus, User
from app.schemas import (
    SessionCreate,
    SessionUpdate,
    SessionResponse,
    SessionDetail,
    SessionListResponse,
)


router = APIRouter()


# ============================================================================
# Helper Functions
# ============================================================================

def get_session_or_404(db: DBSession, session_id: int) -> Session:
    """
    Get session by ID or raise 404.

    Args:
        db: Database session
        session_id: Session ID

    Returns:
        Session: The session

    Raises:
        HTTPException: 404 if session not found
    """
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    return session


def build_session_response(session: Session, include_counts: bool = True) -> dict:
    """
    Build SessionResponse dict from Session model.

    Args:
        session: Session model instance
        include_counts: Whether to include image/comparison counts

    Returns:
        dict: SessionResponse-compatible dict
    """
    response = {
        "id": session.id,
        "user_id": session.user_id,
        "patient_identifier": session.patient_identifier,
        "status": session.status,
        "notes": session.notes,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }

    if include_counts:
        response["image_count"] = session.image_count
        response["comparison_count"] = session.comparison_count
    else:
        response["image_count"] = None
        response["comparison_count"] = None

    return response


# ============================================================================
# Create Session
# ============================================================================

@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new session",
    description="Create a new ultrasound scanning session"
)
def create_session(
    session_data: SessionCreate,
    db: DBSession = Depends(get_db)
) -> SessionResponse:
    """
    Create a new ultrasound scanning session.

    **Process:**
    1. Creates a new session record
    2. Sets status to ACTIVE by default
    3. Associates with user (currently using default user_id=1)
    4. Returns session metadata

    **Future:**
    - user_id will come from authenticated user (JWT token)
    - For now, defaults to user_id=1

    **Use case:**
    Start a new scanning session when an operator begins working with a patient.
    """
    # TODO: Get user_id from authenticated user (JWT token)
    # For now, use default user_id=1 (created by setup_test_data.py)
    user_id = 1

    # Verify user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User {user_id} not found. Run setup_test_data.py first."
        )

    # Create session
    session = Session(
        user_id=user_id,
        patient_identifier=session_data.patient_identifier,
        status=SessionStatus.ACTIVE,
        notes=session_data.notes,
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    # Build response
    response_data = build_session_response(session)
    return SessionResponse(**response_data)


# ============================================================================
# List Sessions
# ============================================================================

@router.get(
    "",
    response_model=SessionListResponse,
    summary="List sessions",
    description="Get a paginated list of scanning sessions"
)
def list_sessions(
    status_filter: Optional[SessionStatus] = Query(
        None,
        description="Filter by session status"
    ),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: DBSession = Depends(get_db)
) -> SessionListResponse:
    """
    Get a paginated list of scanning sessions.

    **Filters:**
    - status: Filter by session status (active, completed, cancelled)

    **Pagination:**
    - page: Page number (starts at 1)
    - page_size: Number of items per page (max 100)

    **Ordering:**
    - Sessions are ordered by creation time (newest first)

    **Use case:**
    Display a list of recent scanning sessions, optionally filtered by status.
    """
    # Build query
    query = db.query(Session)

    # Apply status filter
    if status_filter:
        query = query.filter(Session.status == status_filter)

    # Get total count (before pagination)
    total = query.count()

    # Apply pagination
    offset = (page - 1) * page_size
    sessions = (
        query
        .order_by(Session.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    # Build response
    session_responses = [
        SessionResponse(**build_session_response(s))
        for s in sessions
    ]

    return SessionListResponse(
        items=session_responses,
        total=total,
        page=page,
        page_size=page_size
    )


# ============================================================================
# Get Session Details
# ============================================================================

@router.get(
    "/{session_id}",
    response_model=SessionDetail,
    summary="Get session details",
    description="Get detailed information about a specific session"
)
def get_session(
    session_id: int,
    include_images: bool = Query(
        False,
        description="Include list of images"
    ),
    include_comparisons: bool = Query(
        False,
        description="Include list of comparisons"
    ),
    db: DBSession = Depends(get_db)
) -> SessionDetail:
    """
    Get detailed information about a specific session.

    **Optional includes:**
    - include_images: Include full list of session images
    - include_comparisons: Include full list of comparisons

    **Use case:**
    Display full session details including uploaded images and analysis results.
    """
    session = get_session_or_404(db, session_id)

    # Build base response
    response_data = build_session_response(session)

    # Include images if requested
    if include_images:
        from app.api.images import build_image_response
        response_data["images"] = [
            build_image_response(img)
            for img in session.images
        ]
    else:
        response_data["images"] = None

    # Include comparisons if requested
    if include_comparisons:
        response_data["comparisons"] = [
            {
                "id": comp.id,
                "image_id": comp.image_id,
                "reference_view_id": comp.reference_view_id,
                "ssim": comp.ssim,
                "ncc": comp.ncc,
                "verdict": comp.verdict.value,
                "message": comp.message,
                "confidence": comp.confidence,
                "created_at": comp.created_at,
            }
            for comp in session.comparisons
        ]
    else:
        response_data["comparisons"] = None

    return SessionDetail(**response_data)


# ============================================================================
# Update Session
# ============================================================================

@router.patch(
    "/{session_id}",
    response_model=SessionResponse,
    summary="Update session",
    description="Update session status or notes"
)
def update_session(
    session_id: int,
    session_update: SessionUpdate,
    db: DBSession = Depends(get_db)
) -> SessionResponse:
    """
    Update a session's status or notes.

    **Updatable fields:**
    - status: Change session status (active, completed, cancelled)
    - notes: Update session notes
    - patient_identifier: Update patient identifier

    **Use case:**
    - Mark session as completed when scanning is done
    - Add notes during or after the session
    - Update patient identifier if needed
    """
    session = get_session_or_404(db, session_id)

    # Update fields that were provided
    if session_update.status is not None:
        session.status = session_update.status

    if session_update.notes is not None:
        session.notes = session_update.notes

    if session_update.patient_identifier is not None:
        session.patient_identifier = session_update.patient_identifier

    db.commit()
    db.refresh(session)

    # Build response
    response_data = build_session_response(session)
    return SessionResponse(**response_data)


# ============================================================================
# Delete Session
# ============================================================================

@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete session",
    description="Delete a session and all associated data"
)
def delete_session(
    session_id: int,
    db: DBSession = Depends(get_db)
) -> None:
    """
    Delete a session and all associated data.

    **This will CASCADE delete:**
    - All images in this session
    - All comparisons in this session
    - The session directory and all files

    **Warning:** This action cannot be undone.

    **Use case:**
    Clean up test sessions or remove sessions that were created by mistake.
    """
    session = get_session_or_404(db, session_id)

    # Delete all image files from storage
    try:
        delete_session_directory(session_id)
    except Exception as e:
        # Log error but continue with database deletion
        print(f"Warning: Failed to delete session directory: {e}")

    # Delete database record (will CASCADE delete images and comparisons)
    db.delete(session)
    db.commit()

    # Return 204 No Content
    return None
