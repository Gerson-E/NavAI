# Testing Guide - Precision Probe

## Quick Start Testing

### Step 1: Start the Backend Server

Open a terminal and run:

```bash
cd Backend
python3.11 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### Step 2: Start the Frontend

Open a **new terminal** (keep the backend running) and run:

```bash
cd frontend
npm run dev
```

You should see:
```
  VITE v7.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

### Step 3: Open the Application

Open your browser and go to: **http://localhost:5173**

---

## Testing Checklist

### ✅ 1. Session Management

1. **View Existing Sessions**
   - You should see the test session "TEST-001" in the sidebar
   - It should show "0 images" and "0 analyses"

2. **Create a New Session**
   - Click "+ New Session" button
   - Optionally enter:
     - Patient ID: e.g., "PATIENT-123"
     - Notes: e.g., "Cardiac screening"
   - Click "Create Session"
   - The new session should appear in the list
   - It should automatically be selected

3. **Select a Session**
   - Click on any session in the list
   - It should highlight as "active"
   - The main content area should update

### ✅ 2. Image Upload

1. **Upload via Click**
   - Click the upload area
   - Select a PNG or JPEG image from your computer
   - You should see:
     - "Uploading image..." with spinner
     - Then "Upload successful!" with checkmark
     - The image should appear in the gallery

2. **Upload via Drag & Drop**
   - Drag an image file over the upload area
   - The area should highlight (blue border)
   - Drop the image
   - Same success feedback as above

3. **Test Validation**
   - Try uploading a non-image file → Should show error
   - Try uploading a file > 10MB → Should show error

### ✅ 3. Image Gallery

1. **View Uploaded Images**
   - After uploading, images appear in the left sidebar
   - Click on an image thumbnail to view it
   - The full image should display in the main area

2. **Image Information**
   - Filename should be shown
   - Dimensions and file size should be displayed

### ✅ 4. Position Analysis

1. **Select Reference View**
   - Choose a reference view from the dropdown:
     - Kidney Longitudinal
     - Cardiac 4-Chamber
     - Cardiac Parasternal Long
     - Liver Standard

2. **Run Analysis**
   - Click "Check Position" button
   - You should see "Analyzing..." state
   - Results should appear with:
     - **Verdict badge** (Good/Borderline/Poor) with color:
       - 🟢 Green = Good (SSIM > 0.75)
       - 🟡 Yellow = Borderline (0.5 < SSIM ≤ 0.75)
       - 🔴 Red = Poor (SSIM ≤ 0.5)
     - **Scores**: SSIM, NCC, Confidence
     - **Message**: Feedback from the analysis engine
     - **Processing time**

### ✅ 5. Organ Classification

1. **Classify Organ**
   - Click "Classify Organ" button
   - You should see "Classifying..." state
   - Results should show:
     - **Detected Organ**: e.g., "kidney", "liver", etc.
     - **Confidence**: Percentage
     - **Kidney Badge**: If kidney is detected
     - **Message**: Classification feedback

### ✅ 6. Multiple Images

1. **Upload Multiple Images**
   - Upload 2-3 different images
   - All should appear in the gallery
   - Click between them to switch views
   - Each can be analyzed independently

2. **Session Statistics**
   - Check the session card in sidebar
   - Image count and analysis count should update

---

## Troubleshooting

### Backend Not Starting

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
cd Backend
pip install -r requirements.txt
```

### Database Issues

**Error**: `no such table: users`

**Solution**:
```bash
cd Backend
python3.11 -c "from app.models import *; from app.core.database import create_tables; create_tables()"
python3.11 setup_test_data.py
```

### Frontend Not Connecting

**Error**: "Failed to fetch" or "Cannot connect to backend"

**Check**:
1. Is backend running on port 8000?
2. Check browser console (F12) for errors
3. Verify API URL in frontend `.env` file (if you created one)

### CORS Errors

**Error**: CORS policy blocking requests

**Solution**: Backend already has CORS enabled for all origins. If you see this, restart the backend server.

---

## Expected Behavior

### Successful Upload Flow:
1. Select/Create session
2. Upload image → See spinner → See success checkmark
3. Image appears in gallery
4. Click image to view
5. Run analysis → See results with color-coded verdict

### Analysis Results Format:
- **Good**: Green badge, SSIM typically > 0.75
- **Borderline**: Yellow badge, SSIM 0.5-0.75
- **Poor**: Red badge, SSIM < 0.5

---

## Test Images

You can use any PNG or JPEG images for testing. For best results with the analysis engine, use:
- Ultrasound images (kidney, cardiac, liver)
- Medical imaging samples
- Any image to test the upload/display functionality

---

## API Testing (Optional)

You can also test the API directly:

```bash
# Health check
curl http://localhost:8000/health

# List sessions
curl http://localhost:8000/api/v1/sessions

# API Documentation
open http://localhost:8000/docs
```

---

## Next Steps

Once everything works:
1. ✅ Upload images
2. ✅ View images in gallery
3. ✅ Run position analysis
4. ✅ Run organ classification
5. ✅ Create multiple sessions
6. ✅ Test with different reference views

Enjoy testing! 🎉

