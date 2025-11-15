"""
Setup test data for testing the API.

Creates:
- A test user
- A test session

Run this before testing the image upload endpoint.

Usage:
    python setup_test_data.py
"""

from app.core.database import SessionLocal
from app.models import User, Session, SessionStatus

def setup_test_data():
    """Create test user and session."""
    db = SessionLocal()

    try:
        # Check if test user already exists
        user = db.query(User).filter(User.email == "test@example.com").first()

        if not user:
            # Create test user
            user = User(
                email="test@example.com",
                hashed_password="fake_hash_for_testing",
                full_name="Test User",
                is_active=True
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"✅ Created test user: {user.email} (ID: {user.id})")
        else:
            print(f"✅ Test user already exists: {user.email} (ID: {user.id})")

        # Check if test session already exists
        session = db.query(Session).filter(
            Session.user_id == user.id,
            Session.patient_identifier == "TEST-001"
        ).first()

        if not session:
            # Create test session
            session = Session(
                user_id=user.id,
                patient_identifier="TEST-001",
                status=SessionStatus.ACTIVE,
                notes="Test session for image uploads"
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            print(f"✅ Created test session: {session.id}")
        else:
            print(f"✅ Test session already exists: {session.id}")

        print("\nTest data ready!")
        print(f"User ID: {user.id}")
        print(f"Session ID: {session.id}")
        print("\nYou can now test image upload with:")
        print(f"curl -X POST http://localhost:8000/api/v1/sessions/{session.id}/images \\")
        print(f'  -F "file=@/path/to/image.png"')

    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()

    finally:
        db.close()


if __name__ == "__main__":
    setup_test_data()
