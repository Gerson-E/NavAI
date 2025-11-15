# Person B Implementation Guide - MVP Kidney Detection

## Overview

You are responsible for implementing the **computer vision/machine learning analysis** that powers the kidney detection feature. Person A has built the complete API infrastructure, database, and endpoints. Your job is to replace the stub implementation with a real deep learning model.

---

## 🎯 MVP Goal

Build a classifier that can **detect whether an ultrasound image shows a kidney or not**.

**Success criteria:**
- Take an ultrasound image path as input
- Return whether it's a kidney with confidence score
- Must follow the interface contract (see below)

---

## 📁 File Structure - What You Own

### Your Code (Person B owns these):
```
app/analysis/
├── engine.py          ← YOU IMPLEMENT THIS (replace stub)
└── interface.py       ← THE CONTRACT (don't modify without agreement)
```

### Person A's Code (DO NOT MODIFY):
```
app/
├── api/               ← Person A's API layer
├── core/              ← Person A's database/config
├── models/            ← Person A's database models
└── schemas/           ← Person A's request/response schemas
```

**RULE:** Only edit files in `app/analysis/engine.py`. Everything else is Person A's responsibility.

---

## 🔌 The Interface Contract

**File:** `app/analysis/interface.py`

This is THE CONTRACT between you and Person A. Both of you must agree before changing this file.

### Function You Must Implement

```python
def classify_organ(img_path: str) -> ClassificationResult:
    """
    Classify what organ is shown in an ultrasound image.

    Args:
        img_path: Absolute path to ultrasound image file

    Returns:
        ClassificationResult with these EXACT fields:
        {
            "detected_organ": str,    # "kidney", "liver", "heart", etc.
            "confidence": float,       # 0.0 to 1.0
            "is_kidney": bool,         # True if kidney detected
            "message": str             # Human-readable feedback
        }

    Raises:
        FileNotFoundError: If img_path doesn't exist
        ValueError: If image is invalid/corrupted
        RuntimeError: If classification fails
    """
```

### Return Format (TypedDict)

```python
class ClassificationResult(TypedDict):
    detected_organ: str      # What you detected: "kidney", "liver", "heart", "bladder", "unknown"
    confidence: float        # 0.0 to 1.0 (model confidence)
    is_kidney: bool          # True if detected_organ == "kidney"
    message: str             # Human-readable message for operator
```

---

## 📝 Implementation Steps for MVP

### Step 1: Prepare Your Model

**Options:**
1. **Transfer Learning** (Recommended for MVP)
   - Use pre-trained ResNet/EfficientNet
   - Fine-tune on ultrasound kidney dataset
   - Binary classifier: kidney vs. not-kidney

2. **Custom CNN**
   - Build from scratch if you have large dataset
   - More control but requires more data

3. **Pre-trained Medical Model**
   - Look for existing ultrasound models
   - Medical imaging models on HuggingFace/TensorFlow Hub

**Recommended MVP approach:**
```python
# Example: Using a simple binary classifier
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

# Load your trained model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Binary: kidney or not
model.load_state_dict(torch.load('models/kidney_classifier.pth'))
model.eval()
```

### Step 2: Implement classify_organ()

**File:** `app/analysis/engine.py`

Replace the stub with your implementation:

```python
"""
Image Analysis Engine - Person B's implementation.
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from app.analysis.interface import ClassificationResult

# Load your model once at module level
MODEL_PATH = "models/kidney_classifier.pth"
model = None  # Load your model here

def load_model():
    """Load the kidney classification model."""
    global model
    if model is None:
        # TODO: Load your trained model
        # Example:
        # model = torch.load(MODEL_PATH)
        # model.eval()
        pass
    return model


def classify_organ(img_path: str) -> ClassificationResult:
    """
    Classify organ in ultrasound image.

    MVP: Binary kidney detection.
    """
    # ========================================================================
    # 1. Validation
    # ========================================================================

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # ========================================================================
    # 2. Load and preprocess image
    # ========================================================================

    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")

    # Preprocess for your model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0)

    # ========================================================================
    # 3. Run inference
    # ========================================================================

    model = load_model()

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

        is_kidney = predicted.item() == 1  # Assuming class 1 is kidney
        confidence_score = confidence.item()

    # ========================================================================
    # 4. Format result according to contract
    # ========================================================================

    if is_kidney:
        detected_organ = "kidney"
        if confidence_score >= 0.8:
            message = f"Kidney detected with high confidence ({confidence_score:.2%})"
        elif confidence_score >= 0.6:
            message = f"Kidney detected with moderate confidence ({confidence_score:.2%})"
        else:
            message = f"Possible kidney detected with low confidence ({confidence_score:.2%})"
    else:
        detected_organ = "unknown"  # For MVP, we only classify kidney
        message = f"No kidney detected (confidence: {confidence_score:.2%})"

    # Return in exact format specified by interface
    return ClassificationResult(
        detected_organ=detected_organ,
        confidence=float(confidence_score),
        is_kidney=is_kidney,
        message=message
    )


# Add your helper functions below
def preprocess_ultrasound(img: Image.Image) -> torch.Tensor:
    """Preprocess ultrasound image for model."""
    # Add ultrasound-specific preprocessing
    pass
```

### Step 3: Add Model Files

Create a `models/` directory in the Backend folder:

```bash
Backend/
├── models/
│   ├── kidney_classifier.pth        # Your trained model weights
│   ├── model_config.json            # Model configuration
│   └── README.md                     # Model documentation
├── app/
│   └── analysis/
│       └── engine.py                 # Your implementation
```

**Important:** Add `models/` to `.gitignore` if model files are large. Use Git LFS or external storage.

---

## 🧪 Testing Your Implementation

### 1. Unit Test Your Function

Create `tests/test_classification.py`:

```python
import pytest
from app.analysis.engine import classify_organ

def test_classify_kidney_image():
    """Test kidney detection on sample image."""
    result = classify_organ("test_data/kidney_sample.png")

    # Validate return format
    assert "detected_organ" in result
    assert "confidence" in result
    assert "is_kidney" in result
    assert "message" in result

    # Validate types
    assert isinstance(result["detected_organ"], str)
    assert isinstance(result["confidence"], float)
    assert isinstance(result["is_kidney"], bool)
    assert isinstance(result["message"], str)

    # Validate ranges
    assert 0.0 <= result["confidence"] <= 1.0

    # For a kidney image, should detect kidney
    assert result["is_kidney"] == True
    assert result["detected_organ"] == "kidney"


def test_classify_non_kidney_image():
    """Test that non-kidney images are rejected."""
    result = classify_organ("test_data/liver_sample.png")

    assert result["is_kidney"] == False
    assert result["detected_organ"] != "kidney"


def test_invalid_image_path():
    """Test error handling for missing file."""
    with pytest.raises(FileNotFoundError):
        classify_organ("nonexistent.png")


def test_corrupted_image():
    """Test error handling for corrupted image."""
    with pytest.raises(ValueError):
        classify_organ("test_data/corrupted.txt")
```

Run tests:
```bash
pytest tests/test_classification.py -v
```

### 2. Test Through the API

Person A has already set up the endpoint. Test your implementation:

```bash
# 1. Start the server
uvicorn app.main:app --reload

# 2. Upload a test image
curl -X POST http://localhost:8000/api/v1/sessions/1/images \
  -F "file=@test_kidney_image.png"

# 3. Classify the image
curl -X POST http://localhost:8000/api/v1/classify-organ \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": 1,
    "image_id": 1
  }'

# Expected response:
# {
#   "id": 1,
#   "session_id": 1,
#   "image_id": 1,
#   "detected_organ": "kidney",
#   "confidence": 0.94,
#   "is_kidney": true,
#   "message": "Kidney detected with high confidence (94.00%)",
#   "processing_time_ms": 145,
#   "created_at": "2025-11-15T22:00:00"
# }
```

---

## 📊 Model Performance Targets for MVP

### Minimum Acceptable Performance:
- **Accuracy:** ≥ 85% on validation set
- **Precision (kidney):** ≥ 80% (avoid false positives)
- **Recall (kidney):** ≥ 80% (avoid missing kidneys)
- **Inference time:** < 500ms per image

### Good Performance:
- **Accuracy:** ≥ 90%
- **Precision:** ≥ 85%
- **Recall:** ≥ 85%
- **Inference time:** < 200ms

---

## 🔄 Integration Workflow

### How Person A Calls Your Code

```python
# In app/api/analysis.py (Person A's code)
from app.analysis.engine import classify_organ

# Person A calls your function like this:
result = classify_organ(image_path)

# Your function MUST return a dict with these exact keys:
# {
#     "detected_organ": "kidney",
#     "confidence": 0.92,
#     "is_kidney": True,
#     "message": "Kidney detected with high confidence"
# }
```

Person A's code then:
1. Validates your response matches the schema
2. Saves it to the database
3. Returns it to the frontend

**You don't need to worry about:**
- Database operations
- API routing
- Request/response formatting
- File storage
- Error handling for API layer

**You only need to worry about:**
- Loading the image
- Running your model
- Returning the correct format

---

## 🚀 Deployment Checklist

Before merging your code to main:

- [ ] Replace stub in `app/analysis/engine.py` with real implementation
- [ ] Add your trained model files to `models/` directory
- [ ] Add model loading code to `engine.py`
- [ ] Test with at least 10 different ultrasound images
- [ ] Verify return format matches `ClassificationResult` exactly
- [ ] Handle all error cases (missing file, corrupted image, etc.)
- [ ] Document any dependencies in `requirements.txt`:
  ```
  torch>=2.0.0
  torchvision>=0.15.0
  Pillow>=9.0.0
  ```
- [ ] Test inference time is < 500ms
- [ ] Create a `models/README.md` explaining your model
- [ ] Run all tests: `pytest tests/`
- [ ] Commit ONLY `app/analysis/engine.py` and `models/`
- [ ] Create PR for Person A to review

---

## 🎓 Training Data Recommendations

### Datasets to Consider:

1. **Kidney Ultrasound Datasets:**
   - Look for public medical imaging datasets on:
     - Kaggle
     - Grand Challenge
     - NIH Data Commons
     - Academic hospital partnerships

2. **Data Augmentation:**
   ```python
   transforms.Compose([
       transforms.RandomRotation(15),
       transforms.RandomHorizontalFlip(),
       transforms.ColorJitter(brightness=0.2, contrast=0.2),
       transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
   ])
   ```

3. **Minimum Dataset Size:**
   - **For MVP:** 500-1000 labeled images (250-500 kidney, 250-500 other organs)
   - **For Production:** 5000+ labeled images

---

## ⚠️ Common Pitfalls to Avoid

### 1. Don't Modify the Interface Without Agreement
```python
# ❌ WRONG - Person A's code will break
def classify_organ(img_path: str) -> dict:
    return {"result": "kidney"}  # Missing required fields!

# ✅ CORRECT - Matches the contract
def classify_organ(img_path: str) -> ClassificationResult:
    return ClassificationResult(
        detected_organ="kidney",
        confidence=0.92,
        is_kidney=True,
        message="Kidney detected"
    )
```

### 2. Don't Modify Person A's Files
```python
# ❌ WRONG - Don't edit Person A's API files
# File: app/api/analysis.py
def classify_organ_endpoint(...):
    # Don't modify this!
```

### 3. Don't Forget Error Handling
```python
# ✅ CORRECT - Handle all errors
def classify_organ(img_path: str) -> ClassificationResult:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    try:
        image = Image.open(img_path)
    except Exception as e:
        raise ValueError(f"Invalid image: {str(e)}")

    # ... rest of implementation
```

---

## 📞 Communication with Person A

### When to Contact Person A:

1. **Interface Changes Needed:**
   - If you need additional fields in `ClassificationResult`
   - If you need to change function signature
   - Example: "Can we add a `raw_scores` field for debugging?"

2. **Performance Issues:**
   - If inference takes > 500ms consistently
   - If you need GPU support

3. **Error Cases:**
   - If you discover new error scenarios not covered by the contract

### When You Don't Need to Contact Person A:

1. **Model Architecture Changes:**
   - Switching from ResNet to EfficientNet
   - Changing training approach
   - As long as the interface stays the same

2. **Preprocessing Changes:**
   - Image normalization
   - Augmentation strategies
   - As long as output format is correct

---

## 📈 Future Enhancements (Beyond MVP)

After MVP is working, you can enhance to:

1. **Multi-class Classification:**
   - Detect kidney, liver, heart, bladder, etc.
   - Update `detected_organ` to return specific organ names

2. **Segmentation:**
   - Return bounding box or mask of kidney location
   - Requires interface update (coordinate with Person A)

3. **Abnormality Detection:**
   - Detect kidney stones, cysts, tumors
   - Requires new fields in interface

4. **Confidence Thresholds:**
   - Add configurable thresholds for "good", "borderline", "poor"

---

## 🔗 Quick Reference

### Your Files:
- `app/analysis/engine.py` - Your implementation
- `models/` - Your model weights

### Contract:
- `app/analysis/interface.py` - THE CONTRACT (read-only unless both agree)

### Testing:
```bash
# Unit tests
pytest tests/test_classification.py

# API test
curl -X POST http://localhost:8000/api/v1/classify-organ \
  -H "Content-Type: application/json" \
  -d '{"session_id": 1, "image_id": 1}'
```

### Dependencies:
```bash
# Add to requirements.txt
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
```

---

## ✅ Success Criteria Summary

Your implementation is complete when:

1. ✅ `classify_organ()` returns correct `ClassificationResult` format
2. ✅ Kidney detection accuracy ≥ 85%
3. ✅ Inference time < 500ms per image
4. ✅ All error cases handled (FileNotFoundError, ValueError, RuntimeError)
5. ✅ Tests pass: `pytest tests/`
6. ✅ API endpoint works: GET `/api/v1/classify-organ` returns valid responses
7. ✅ Model documented in `models/README.md`

---

**Questions?** Contact Person A (Gerson) if you need:
- Interface changes
- New API endpoints
- Database schema modifications
- File storage changes

**Good luck building the MVP!** 🚀
