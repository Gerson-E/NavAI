# NavAI Frontend (minimal)

Prereqs:
- Backend running at http://localhost:8000
- If backend blocks CORS, enable CORS on backend for the frontend origin (typically http://localhost:5173). Do not modify backend here if you don't control it.

Quick start:
1. cd frontend
2. cp .env.example .env (edit VITE_API_BASE if needed)
3. npm install
4. npm run dev
5. Open the dev URL shown by Vite (default http://localhost:5173)

Notes:
- The app posts file uploads to POST /api/v1/sessions/{session_id}/images (multipart/form-data field name "file").
- Analysis endpoints:
  - POST /api/v1/classify-organ with JSON { session_id, image_id }
  - POST /api/v1/check-position with JSON { session_id, image_id, reference_view_id }
- API base can be changed via VITE_API_BASE.
