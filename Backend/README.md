# Ultrasound Positioning Platform - Backend

Backend service for an ultrasound operator guidance platform that provides real-time feedback on probe positioning.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the development server
uvicorn app.main:app --reload

# Run tests
pytest
```

API Documentation: http://localhost:8000/docs

## Team Structure

This backend is built by **2 developers** working in parallel:

- **Person A (Gerson)**: API & Data Layer
- **Person B**: Image Analysis Engine

**READ THIS FIRST:** See [COLLABORATION.md](./COLLABORATION.md) for detailed collaboration guidelines.

## Project Structure

```
Backend/
├── app/
│   ├── main.py                 # FastAPI app (Person A)
│   ├── api/                    # API routes (Person A)
│   ├── models/                 # Database models (Person A)
│   ├── schemas/                # Pydantic schemas (Person A)
│   ├── core/                   # Config & settings (Person A)
│   └── analysis/               # Image analysis (Person B)
│       ├── interface.py        # THE CONTRACT (both teams agree)
│       └── engine.py           # Analysis implementation (Person B)
├── tests/
│   ├── test_api.py             # API tests (Person A)
│   └── test_analysis.py        # Analysis tests (Person B)
├── requirements.txt            # All dependencies
├── requirements-api.txt        # Person A dependencies
└── requirements-analysis.txt   # Person B dependencies
```

## The Interface Contract

The two teams coordinate through a single interface defined in `app/analysis/interface.py`:

```python
def compare_to_reference(current_img_path: str, ref_id: str) -> ComparisonResult
```

**Person A** calls this function from API endpoints.
**Person B** implements the image analysis logic.

See [COLLABORATION.md](./COLLABORATION.md) for the full contract and rules.

## Current Status

- ✅ Project structure established
- ✅ Interface contract defined
- ✅ Stub implementation provided (Person B: replace this)
- 🚧 API endpoints (Person A: in progress)
- 🚧 Image analysis (Person B: in progress)

## Development Workflow

### Person A Workflow
1. Work in branch `person-a/api-layer`
2. Use stub implementation for testing
3. Build API endpoints, database, storage
4. Merge to `main` frequently

### Person B Workflow
1. Work in branch `person-b/analysis-engine`
2. Replace stub in `app/analysis/engine.py`
3. Build preprocessing, metrics, ML models
4. Merge to `main` frequently

### Testing Independently

**Person A** can test API logic by mocking the analysis engine:
```python
@patch('app.analysis.engine.compare_to_reference')
def test_my_endpoint(mock_compare):
    mock_compare.return_value = {"ssim": 0.8, ...}
    # Test API logic
```

**Person B** can test analysis without the API:
```python
def test_analysis():
    result = compare_to_reference("/path/to/test.png", "ref_id")
    assert result["ssim"] > 0
```

## Tech Stack

### API Layer (Person A)
- FastAPI
- SQLAlchemy (ORM)
- Alembic (migrations)
- Pydantic (validation)

### Analysis Engine (Person B)
- NumPy
- OpenCV
- scikit-image (SSIM)
- Pillow

## Next Steps

1. **Both**: Review and agree on `app/analysis/interface.py`
2. **Person B**: Provide stub implementation → Done ✅
3. **Person A**: Build API using stub
4. **Person B**: Build real analysis engine
5. **Both**: Integration testing
6. **Both**: Merge to main

## Questions?

See [COLLABORATION.md](./COLLABORATION.md) or discuss with your partner.
