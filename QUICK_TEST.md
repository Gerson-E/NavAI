# Quick Testing Guide - Position Analysis

## Method 1: Test from Frontend (Easiest)

### Step 1: Make sure both servers are running

**Terminal 1 - Backend:**
```bash
cd Backend
python3.11 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Step 2: Open the app
Go to: http://localhost:5173 (or the port shown in terminal)

### Step 3: Test the analysis
1. **Select a session** (or create a new one)
2. **Upload an image** (drag & drop or click)
3. **Click on the uploaded image** in the gallery
4. **Select a reference view** from the dropdown (e.g., "Kidney Longitudinal")
5. **Click "Check Position"**
6. **Check the results** - You should see:
   - Green/Yellow/Red verdict badge
   - SSIM and NCC scores
   - Feedback message

### Step 4: Check for errors
- **Browser Console (F12)**: Look for any red error messages
- **Backend Terminal**: Check for Python errors
- **Alert popup**: Should show detailed error if something fails

---

## Method 2: Test API Directly (Advanced)

### Test with curl

```bash
# First, find your session and image IDs
curl http://localhost:8000/api/v1/sessions | python3.11 -m json.tool

# Then test the analysis endpoint
curl -X POST http://localhost:8000/api/v1/check-position \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": 4,
    "image_id": 3,
    "reference_view_id": "kidney_longitudinal"
  }' | python3.11 -m json.tool
```

### Expected Response:
```json
{
  "id": 2,
  "session_id": 4,
  "image_id": 3,
  "reference_view_id": "kidney_longitudinal",
  "ssim": 0.78,
  "ncc": 0.72,
  "verdict": "good",
  "message": "[STUB] Probe positioning looks good for kidney_longitudinal. This is dummy data.",
  "confidence": 0.85,
  "processing_time_ms": 0,
  "created_at": "2025-11-24T01:22:37"
}
```

---

## Method 3: Test from Python (For debugging)

```python
# test_analysis.py
from app.analysis.engine import compare_to_reference
from pathlib import Path

# Find an uploaded image
image_path = Path("Backend/media/sessions/4/20251123_194717_80a77132_kidney1.png").absolute()

if image_path.exists():
    result = compare_to_reference(str(image_path), "kidney_longitudinal")
    print("✅ Success!")
    print(f"SSIM: {result['ssim']}")
    print(f"NCC: {result['ncc']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Message: {result['message']}")
else:
    print(f"❌ Image not found: {image_path}")
```

Run it:
```bash
cd Backend
python3.11 test_analysis.py
```

---

## Troubleshooting

### Issue: "Failed to analyze position"
1. **Check browser console (F12)** - Look for the detailed error
2. **Check backend terminal** - Look for Python traceback
3. **Verify image exists**: 
   ```bash
   ls -la Backend/media/sessions/*/
   ```

### Issue: "Image not found"
- The image file might have been deleted
- Try uploading the image again

### Issue: "Session/Image ID mismatch"
- Make sure the image belongs to the selected session
- Check the console logs to see what IDs are being sent

### Issue: Backend not responding
- Make sure backend is running on port 8000
- Test: `curl http://localhost:8000/health`

---

## What You Should See

### ✅ Success:
- Green/Yellow/Red badge with verdict
- SSIM score (0.0 to 1.0)
- NCC score (-1.0 to 1.0)
- Confidence percentage
- Feedback message
- Processing time

### ❌ Error:
- Detailed error message in alert
- Error details in browser console
- Backend error in terminal

---

## Quick Checklist

- [ ] Backend running on port 8000
- [ ] Frontend running (check terminal for URL)
- [ ] Session created/selected
- [ ] Image uploaded successfully
- [ ] Image selected in gallery
- [ ] Reference view selected
- [ ] Clicked "Check Position"
- [ ] Results displayed OR error message shown

---

## Need Help?

If it's still not working:
1. Copy the exact error message from the alert
2. Copy any errors from browser console (F12)
3. Copy any errors from backend terminal
4. Share those details for debugging!

