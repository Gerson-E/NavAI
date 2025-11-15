"""
Seed reference views into the database.

Creates the initial set of reference ultrasound views that images can be compared against.

Run this before testing the check-position endpoint.

Usage:
    python seed_reference_views.py
"""

from app.core.database import SessionLocal
from app.models import ReferenceView, ReferenceViewCategory

def seed_reference_views():
    """Create initial reference views."""
    db = SessionLocal()

    # Define reference views
    reference_views = [
        {
            "id": "cardiac_4chamber",
            "name": "Cardiac 4-Chamber View",
            "description": "Standard apical 4-chamber cardiac view showing all four chambers of the heart",
            "category": ReferenceViewCategory.CARDIAC,
            "storage_path": "reference_views/cardiac_4chamber.png",
            "sort_order": 1,
        },
        {
            "id": "cardiac_parasternal_long",
            "name": "Cardiac Parasternal Long Axis",
            "description": "Long-axis view of the left ventricle and left atrium",
            "category": ReferenceViewCategory.CARDIAC,
            "storage_path": "reference_views/cardiac_parasternal_long.png",
            "sort_order": 2,
        },
        {
            "id": "liver_standard",
            "name": "Standard Liver View",
            "description": "Standard liver parenchyma view",
            "category": ReferenceViewCategory.ABDOMINAL,
            "storage_path": "reference_views/liver_standard.png",
            "sort_order": 10,
        },
        {
            "id": "kidney_longitudinal",
            "name": "Kidney Longitudinal View",
            "description": "Longitudinal view of the kidney",
            "category": ReferenceViewCategory.ABDOMINAL,
            "storage_path": "reference_views/kidney_longitudinal.png",
            "sort_order": 11,
        },
    ]

    try:
        for ref_data in reference_views:
            # Check if already exists
            existing = db.query(ReferenceView).filter(
                ReferenceView.id == ref_data["id"]
            ).first()

            if existing:
                print(f"✅ Reference view already exists: {ref_data['id']}")
            else:
                # Create new reference view
                ref_view = ReferenceView(**ref_data)
                db.add(ref_view)
                print(f"✅ Created reference view: {ref_data['id']}")

        db.commit()
        print("\n✅ Reference views seeded successfully!")
        print("\nAvailable reference views:")
        for ref in reference_views:
            print(f"  - {ref['id']}: {ref['name']}")

        print("\nYou can now test the check-position endpoint with:")
        print('curl -X POST http://localhost:8000/api/v1/check-position \\')
        print('  -H "Content-Type: application/json" \\')
        print("  -d '{")
        print('    "session_id": 1,')
        print('    "image_id": 1,')
        print('    "reference_view_id": "cardiac_4chamber"')
        print("  }'")

    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()

    finally:
        db.close()


if __name__ == "__main__":
    seed_reference_views()
