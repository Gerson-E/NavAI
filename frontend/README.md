# Precision Probe - Frontend

A modern React frontend for the Precision Probe ultrasound image analysis platform.

## Features

- **Session Management**: Create and manage ultrasound scanning sessions
- **Image Upload**: Drag-and-drop or click to upload ultrasound images (PNG/JPEG, max 10MB)
- **Position Analysis**: Compare uploaded images against reference views to check probe positioning
- **Organ Classification**: Detect and classify organs in ultrasound images (MVP: kidney detection)
- **Real-time Feedback**: Get immediate feedback with color-coded results and detailed scores

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend API running on http://localhost:8000 (or configure via environment variable)

### Installation

1. Install dependencies:
```bash
npm install
```

2. (Optional) Configure API URL:
```bash
cp .env.example .env
# Edit .env to set VITE_API_BASE_URL if your backend is on a different port
```

3. Start the development server:
```bash
npm run dev
```

The frontend will be available at http://localhost:5173 (or the port shown in the terminal).

### Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Project Structure

```
src/
  ├── api/
  │   └── client.ts          # API client and type definitions
  ├── components/
  │   ├── SessionManager.tsx # Session creation and selection
  │   ├── ImageUpload.tsx    # Image upload with drag-and-drop
  │   ├── ImageDisplay.tsx    # Image display and analysis controls
  │   └── ImageGallery.tsx   # Image gallery and navigation
  ├── App.tsx                 # Main application component
  └── index.css              # Global styles
```

## Usage

1. **Create a Session**: Click "New Session" to create a new scanning session (optionally add patient ID and notes)

2. **Upload Images**: Drag and drop or click to upload ultrasound images

3. **Analyze Images**:
   - **Position Analysis**: Select a reference view and click "Check Position" to compare your image
   - **Organ Classification**: Click "Classify Organ" to detect what organ is shown

4. **View Results**: Results are displayed with color-coded verdicts:
   - 🟢 **Good**: SSIM > 0.75
   - 🟡 **Borderline**: 0.5 < SSIM ≤ 0.75
   - 🔴 **Poor**: SSIM ≤ 0.5

## Technologies

- **React 18** with TypeScript
- **Vite** for fast development and building
- **Axios** for API communication
- **CSS3** for styling (no external UI libraries)

## API Integration

The frontend communicates with the backend API at the following endpoints:

- `POST /api/v1/sessions` - Create session
- `GET /api/v1/sessions` - List sessions
- `POST /api/v1/sessions/{id}/images` - Upload image
- `GET /api/v1/sessions/{id}/images` - List session images
- `POST /api/v1/check-position` - Analyze position
- `POST /api/v1/classify-organ` - Classify organ

See `src/api/client.ts` for the complete API client implementation.
