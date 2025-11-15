# Backend Collaboration Guide

## Team Split

### Person A: API & Data Layer (Gerson)
**Owns:** FastAPI routes, database models, storage, authentication

**Your directories:**
- `app/api/` - All API route handlers
- `app/models/` - Database models (SQLAlchemy)
- `app/schemas/` - Pydantic request/response schemas
- `app/core/` - Config, settings, dependencies
- `app/main.py` - FastAPI app setup

**DO NOT TOUCH:** `app/analysis/` (except importing from `interface.py` and `engine.py`)

### Person B: Image Analysis Engine
**Owns:** Image processing, similarity metrics, ML/CV algorithms

**Your directories:**
- `app/analysis/` - All image processing and analysis code

**DO NOT TOUCH:** `app/api/`, `app/models/`, `app/schemas/`, `app/core/`, `app/main.py`

## The Sacred Contract: `app/analysis/interface.py`

This file is the **interface contract** between both teams.

**Rules:**
1. Never modify without discussing with your partner first
2. The function signature `compare_to_reference()` must remain stable
3. Changes to `ComparisonResult` fields require both teams to agree

## Current Setup

### Person B: Your Stub Implementation

There's a stub in `app/analysis/engine.py` that returns dummy data. Replace it with real implementation:

```python
from app.analysis.engine import compare_to_reference

result = compare_to_reference("/path/to/image.png", "cardiac_4chamber")
# Returns: {"ssim": 0.78, "ncc": 0.72, "verdict": "good", ...}
```

**Your tasks:**
1. Implement real SSIM calculation
2. Implement real NCC calculation
3. Add preprocessing pipeline
4. Create reference image loading system
5. Implement threshold-based verdict logic
6. Add unit tests in `tests/test_analysis.py`

### Person A: Your Integration Points

You can start building the API **right now** using the stub:

```python
# In your API endpoint
from app.analysis.engine import compare_to_reference

@router.post("/check-position")
async def check_position(session_id: int, image_id: int, ref_id: str):
    # Get image path from your database
    image_path = get_image_path_from_db(image_id)

    # Call Person B's analysis engine
    result = compare_to_reference(image_path, ref_id)

    # Save result to database and return
    save_comparison_result(session_id, result)
    return result
```

**Your tasks:**
1. Set up database (SQLAlchemy models)
2. Implement image upload endpoint
3. Implement session management
4. Implement `/check-position` endpoint
5. Implement reference views API
6. Add integration tests in `tests/test_api.py`

## Git Workflow

### Branching Strategy
- `main` - Stable code only
- `person-a/api-layer` - Person A's work
- `person-b/analysis-engine` - Person B's work

### Merge Order
1. **First integration:** Person B merges analysis module → then Person A merges API layer
2. **Subsequent work:** Merge small features frequently (daily if possible)

### Before Merging
- Run all tests: `pytest`
- Ensure no conflicts in shared files
- Update this README if you changed the interface

## Running the Application

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the API
```bash
# From Backend/ directory
uvicorn app.main:app --reload
```

Visit: http://localhost:8000/docs for API documentation

### Run Tests
```bash
pytest
pytest tests/test_api.py  # Person A's tests
pytest tests/test_analysis.py  # Person B's tests
```

## Project Structure

```
Backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Person A owns
│   ├── api/                    # Person A owns
│   │   ├── __init__.py
│   │   ├── sessions.py         # TODO: Person A
│   │   ├── images.py           # TODO: Person A
│   │   └── reference_views.py  # TODO: Person A
│   ├── models/                 # Person A owns
│   │   ├── __init__.py
│   │   ├── session.py          # TODO: Person A
│   │   ├── image.py            # TODO: Person A
│   │   └── comparison.py       # TODO: Person A
│   ├── schemas/                # Person A owns
│   │   ├── __init__.py
│   │   └── ...                 # TODO: Person A
│   ├── core/                   # Person A owns
│   │   ├── __init__.py
│   │   ├── config.py           # TODO: Person A
│   │   └── database.py         # TODO: Person A
│   └── analysis/               # Person B owns
│       ├── __init__.py
│       ├── interface.py        # THE CONTRACT (both agree)
│       ├── engine.py           # TODO: Person B (currently stub)
│       ├── preprocessing.py    # TODO: Person B
│       ├── metrics.py          # TODO: Person B
│       └── reference_loader.py # TODO: Person B
├── tests/
│   ├── test_api.py             # Person A
│   └── test_analysis.py        # Person B
├── requirements.txt
├── requirements-api.txt        # Person A's dependencies
├── requirements-analysis.txt   # Person B's dependencies
└── .gitignore
```

## Communication Checklist

### Daily (5-10 min sync)
- [ ] What did you complete yesterday?
- [ ] What are you working on today?
- [ ] Any blockers or interface changes needed?

### Before Changing `interface.py`
- [ ] Discussed with partner
- [ ] Both agreed on the change
- [ ] Updated this README
- [ ] Updated tests on both sides

## Common Scenarios

### Scenario: Person B needs to add a new field to ComparisonResult
1. Person B opens an issue/discussion
2. Both discuss if Person A needs this field
3. If yes, both update their tests
4. Person B makes the change to `interface.py`
5. Person B updates `engine.py` implementation
6. Person A updates API to expose the new field

### Scenario: Person A needs a new analysis function
1. Person A describes the requirement
2. Both agree on the function signature
3. Person B adds it to `interface.py` and implements in `engine.py`
4. Person A integrates it into API

## Questions?

If you're unsure about:
- **"Should I touch this file?"** → Check the ownership in this README
- **"Can I change the interface?"** → Ask your partner first
- **"Tests are failing"** → Check if it's your code or the stub/integration

## Next Steps

### Person A - Week 1 Goals
- [ ] Set up database with SQLAlchemy
- [ ] Implement image upload endpoint (using stub analysis)
- [ ] Implement session management
- [ ] Create basic CRUD for reference views

### Person B - Week 1 Goals
- [ ] Replace stub with real SSIM calculation
- [ ] Replace stub with real NCC calculation
- [ ] Implement image preprocessing pipeline
- [ ] Add unit tests with test images
