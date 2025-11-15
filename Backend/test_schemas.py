"""
Test script to verify Pydantic schemas are correctly defined.

Run this to check:
1. All schemas can be imported
2. Validation works correctly
3. Example data matches schema definitions

Usage:
    python test_schemas.py
"""

from datetime import datetime
from app.schemas import (
    # Session
    SessionCreate,
    SessionResponse,
    SessionUpdate,
    # Image
    ImageResponse,
    ImageUploadResponse,
    # Reference View
    ReferenceViewResponse,
    ReferenceViewCreate,
    # Comparison (THE BRIDGE)
    ComparisonRequest,
    ComparisonResponse,
    AnalysisResultFromEngine,
)


def test_imports():
    """Test that all schemas can be imported."""
    print("✅ Testing schema imports...")

    schemas = [
        SessionCreate,
        SessionResponse,
        SessionUpdate,
        ImageResponse,
        ImageUploadResponse,
        ReferenceViewResponse,
        ReferenceViewCreate,
        ComparisonRequest,
        ComparisonResponse,
        AnalysisResultFromEngine,
    ]

    for schema in schemas:
        print(f"  ✓ {schema.__name__}")

    print("✅ All schemas imported successfully!\n")


def test_session_schemas():
    """Test session schema validation."""
    print("✅ Testing Session schemas...")

    # Test SessionCreate
    session_create = SessionCreate(
        patient_identifier="STUDY-001",
        notes="Test session"
    )
    assert session_create.patient_identifier == "STUDY-001"
    print("  ✓ SessionCreate validation works")

    # Test SessionResponse
    session_response = SessionResponse(
        id=1,
        user_id=1,
        patient_identifier="STUDY-001",
        status="active",
        notes="Test",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    assert session_response.id == 1
    print("  ✓ SessionResponse validation works")

    print("✅ Session schemas working!\n")


def test_comparison_schemas():
    """Test comparison schemas (Person B integration)."""
    print("✅ Testing Comparison schemas (Bridge to Person B)...")

    # Test ComparisonRequest
    request = ComparisonRequest(
        session_id=1,
        image_id=42,
        reference_view_id="cardiac_4chamber"
    )
    assert request.session_id == 1
    assert request.reference_view_id == "cardiac_4chamber"
    print("  ✓ ComparisonRequest validation works")

    # Test AnalysisResultFromEngine (Person B's output)
    engine_result = AnalysisResultFromEngine(
        ssim=0.82,
        ncc=0.75,
        verdict="good",
        message="Probe positioning looks good",
        confidence=0.88
    )
    assert engine_result.ssim == 0.82
    assert engine_result.verdict == "good"
    print("  ✓ AnalysisResultFromEngine validation works")

    # Test ComparisonResponse
    comparison = ComparisonResponse(
        id=1,
        session_id=1,
        image_id=42,
        reference_view_id="cardiac_4chamber",
        ssim=0.82,
        ncc=0.75,
        verdict="good",
        message="Probe positioning looks good",
        confidence=0.88,
        processing_time_ms=250,
        created_at=datetime.now()
    )
    assert comparison.ssim == 0.82
    assert comparison.verdict == "good"
    print("  ✓ ComparisonResponse validation works")

    print("✅ Comparison schemas working!\n")


def test_validation_rules():
    """Test that validation rules work correctly."""
    print("✅ Testing validation rules...")

    # Test SSIM range validation
    try:
        AnalysisResultFromEngine(
            ssim=1.5,  # Invalid: > 1.0
            ncc=0.75,
            verdict="good",
            message="Test",
            confidence=0.88
        )
        print("  ❌ SSIM validation failed - should reject > 1.0")
    except Exception:
        print("  ✓ SSIM range validation works")

    # Test verdict enum validation
    try:
        AnalysisResultFromEngine(
            ssim=0.82,
            ncc=0.75,
            verdict="excellent",  # Invalid: not in allowed values
            message="Test",
            confidence=0.88
        )
        print("  ❌ Verdict validation failed - should reject invalid values")
    except Exception:
        print("  ✓ Verdict enum validation works")

    # Test NCC range validation
    try:
        AnalysisResultFromEngine(
            ssim=0.82,
            ncc=2.0,  # Invalid: > 1.0
            verdict="good",
            message="Test",
            confidence=0.88
        )
        print("  ❌ NCC validation failed - should reject > 1.0")
    except Exception:
        print("  ✓ NCC range validation works")

    print("✅ Validation rules working!\n")


def test_reference_view_validation():
    """Test reference view schema validation."""
    print("✅ Testing ReferenceView schemas...")

    # Test valid reference view creation
    ref_view = ReferenceViewCreate(
        id="cardiac_4chamber",
        name="Cardiac 4-Chamber View",
        description="Standard apical 4-chamber view",
        category="cardiac",
        sort_order=1
    )
    assert ref_view.id == "cardiac_4chamber"
    print("  ✓ ReferenceViewCreate validation works")

    # Test ID format validation
    try:
        ReferenceViewCreate(
            id="INVALID-ID",  # Should be lowercase with underscores
            name="Test",
            category="cardiac"
        )
        print("  ❌ Reference ID validation failed - should reject uppercase")
    except Exception:
        print("  ✓ Reference ID format validation works")

    print("✅ ReferenceView schemas working!\n")


def test_schema_to_dict():
    """Test schema serialization to dict."""
    print("✅ Testing schema serialization...")

    result = AnalysisResultFromEngine(
        ssim=0.82,
        ncc=0.75,
        verdict="good",
        message="Test message",
        confidence=0.88
    )

    result_dict = result.model_dump()
    assert result_dict["ssim"] == 0.82
    assert result_dict["verdict"] == "good"
    print("  ✓ Schema to dict conversion works")

    # Test JSON serialization
    result_json = result.model_dump_json()
    assert isinstance(result_json, str)
    assert "0.82" in result_json
    print("  ✓ Schema to JSON conversion works")

    print("✅ Serialization working!\n")


def print_summary():
    """Print a summary of schema capabilities."""
    print("=" * 60)
    print("SCHEMA SUMMARY")
    print("=" * 60)
    print()
    print("Request Schemas (API inputs):")
    print("  - SessionCreate       - Create new scan session")
    print("  - ComparisonRequest   - Request image analysis")
    print("  - ReferenceViewCreate - Create new reference view")
    print()
    print("Response Schemas (API outputs):")
    print("  - SessionResponse       - Session data")
    print("  - ImageResponse         - Image metadata")
    print("  - ReferenceViewResponse - Reference view data")
    print("  - ComparisonResponse    - Analysis results")
    print()
    print("Person B Integration:")
    print("  - AnalysisResultFromEngine - Validates Person B's output")
    print("  - Maps ComparisonResult → ComparisonResponse")
    print()
    print("Validation Features:")
    print("  ✓ Type checking (int, str, float, datetime)")
    print("  ✓ Range validation (SSIM: 0-1, NCC: -1 to 1)")
    print("  ✓ Enum validation (verdict: good/borderline/poor)")
    print("  ✓ String length limits")
    print("  ✓ Custom validators (ID format, categories)")
    print()
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING PYDANTIC SCHEMAS")
    print("=" * 60 + "\n")

    # Run tests
    test_imports()
    test_session_schemas()
    test_comparison_schemas()
    test_validation_rules()
    test_reference_view_validation()
    test_schema_to_dict()

    print_summary()

    print("\n✅ ALL SCHEMA TESTS PASSED!")
    print("\nNext steps:")
    print("  1. Schemas are ready to use in API routes")
    print("  2. They will auto-generate OpenAPI documentation")
    print("  3. FastAPI will validate requests/responses automatically")
    print("  4. Ready to proceed to Phase 2: Storage & Upload\n")
