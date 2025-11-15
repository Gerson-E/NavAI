"""
Test script to verify database models are correctly defined.

Run this to check:
1. All models can be imported
2. Database tables can be created
3. Relationships are properly configured

Usage:
    python test_models.py
"""

from app.core.database import Base, engine, create_tables
from app.models import (
    User,
    Session,
    SessionStatus,
    Image,
    ReferenceView,
    ReferenceViewCategory,
    Comparison,
    ComparisonVerdict,
)


def test_imports():
    """Test that all models can be imported."""
    print("✅ Testing model imports...")

    models = [User, Session, Image, ReferenceView, Comparison]
    enums = [SessionStatus, ComparisonVerdict, ReferenceViewCategory]

    for model in models:
        print(f"  ✓ {model.__name__}")

    for enum_cls in enums:
        print(f"  ✓ {enum_cls.__name__}")

    print("✅ All models imported successfully!\n")


def test_table_names():
    """Test that table names are correctly defined."""
    print("✅ Testing table names...")

    tables = {
        User: "users",
        Session: "sessions",
        Image: "images",
        ReferenceView: "reference_views",
        Comparison: "comparisons",
    }

    for model, expected_name in tables.items():
        actual_name = model.__tablename__
        assert actual_name == expected_name, f"Expected {expected_name}, got {actual_name}"
        print(f"  ✓ {model.__name__} -> {actual_name}")

    print("✅ All table names correct!\n")


def test_create_tables():
    """Test that tables can be created."""
    print("✅ Testing table creation...")

    try:
        create_tables()
        print("  ✓ Tables created successfully")

        # Print all created tables
        print(f"\n  Created {len(Base.metadata.tables)} tables:")
        for table_name in Base.metadata.tables.keys():
            print(f"    - {table_name}")

        print("\n✅ Database schema created successfully!\n")
        return True
    except Exception as e:
        print(f"❌ Failed to create tables: {e}")
        return False


def test_relationships():
    """Test that relationships are properly configured."""
    print("✅ Testing model relationships...")

    # Check User relationships
    assert hasattr(User, 'sessions'), "User should have 'sessions' relationship"
    print("  ✓ User.sessions")

    # Check Session relationships
    assert hasattr(Session, 'user'), "Session should have 'user' relationship"
    assert hasattr(Session, 'images'), "Session should have 'images' relationship"
    assert hasattr(Session, 'comparisons'), "Session should have 'comparisons' relationship"
    print("  ✓ Session.user")
    print("  ✓ Session.images")
    print("  ✓ Session.comparisons")

    # Check Image relationships
    assert hasattr(Image, 'session'), "Image should have 'session' relationship"
    assert hasattr(Image, 'comparisons'), "Image should have 'comparisons' relationship"
    print("  ✓ Image.session")
    print("  ✓ Image.comparisons")

    # Check ReferenceView relationships
    assert hasattr(ReferenceView, 'comparisons'), "ReferenceView should have 'comparisons' relationship"
    print("  ✓ ReferenceView.comparisons")

    # Check Comparison relationships
    assert hasattr(Comparison, 'session'), "Comparison should have 'session' relationship"
    assert hasattr(Comparison, 'image'), "Comparison should have 'image' relationship"
    assert hasattr(Comparison, 'reference_view'), "Comparison should have 'reference_view' relationship"
    print("  ✓ Comparison.session")
    print("  ✓ Comparison.image")
    print("  ✓ Comparison.reference_view")

    print("✅ All relationships configured correctly!\n")


def print_summary():
    """Print a summary of the database schema."""
    print("=" * 60)
    print("DATABASE SCHEMA SUMMARY")
    print("=" * 60)
    print()
    print("Tables created:")
    print("  1. users         - User accounts")
    print("  2. sessions      - Scan sessions")
    print("  3. images        - Uploaded ultrasound images")
    print("  4. reference_views - Reference ultrasound views")
    print("  5. comparisons   - Image analysis results")
    print()
    print("Relationships:")
    print("  User 1:N Session")
    print("  Session 1:N Image")
    print("  Session 1:N Comparison")
    print("  Image 1:N Comparison")
    print("  ReferenceView 1:N Comparison")
    print()
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING DATABASE MODELS")
    print("=" * 60 + "\n")

    # Run tests
    test_imports()
    test_table_names()
    test_relationships()
    success = test_create_tables()

    if success:
        print_summary()
        print("\n✅ ALL TESTS PASSED!")
        print("\nNext steps:")
        print("  1. Check navai.db was created")
        print("  2. Install DB browser: brew install --cask db-browser-for-sqlite")
        print("  3. Open navai.db to inspect schema")
        print("  4. Proceed to Part 3: Pydantic Schemas\n")
    else:
        print("\n❌ TESTS FAILED - Check errors above\n")
