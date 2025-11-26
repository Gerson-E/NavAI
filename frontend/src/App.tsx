import { useState, useRef } from 'react';
import { type Session, type Image } from './api/client';
import SessionManager from './components/SessionManager';
import ImageUpload from './components/ImageUpload';
import ImageGallery from './components/ImageGallery';
import ErrorBoundary from './components/ErrorBoundary';
import './App.css';

function App() {
  const [currentSession, setCurrentSession] = useState<Session | null>(null);
  const galleryRefreshRef = useRef<() => void>(() => {});

  const handleImageUploaded = (_image: Image) => {
    // Trigger gallery refresh
    galleryRefreshRef.current();
  };

  return (
    <ErrorBoundary>
      <div className="app">
        <header className="app-header">
          <h1>Precision Probe</h1>
          <p className="subtitle">Learn to take the best ultrasound images with AI-powered feedback</p>
        </header>

        <main className="app-main">
          <div className="app-sidebar">
            <SessionManager
              currentSession={currentSession}
              onSessionChange={setCurrentSession}
            />
          </div>

          <div className="app-content">
            {!currentSession ? (
              <div className="welcome-screen">
                <div className="welcome-content">
                  <h2>Welcome to Precision Probe</h2>
                  <p>Create or select a session to get started</p>
                  <p className="welcome-description">
                    Upload ultrasound images and receive real-time feedback on positioning and image quality.
                    Our AI analysis helps you improve your scanning technique.
                  </p>
                </div>
              </div>
            ) : (
              <>
                <div className="content-section">
                  <h2>Upload Image</h2>
                  <ImageUpload
                    sessionId={currentSession.id}
                    onImageUploaded={handleImageUploaded}
                  />
                </div>

                <div className="content-section">
                  <h2>Images & Analysis</h2>
                  <ImageGallery
                    sessionId={currentSession.id}
                    onRefreshRef={galleryRefreshRef}
                  />
                </div>
              </>
            )}
          </div>
        </main>
      </div>
    </ErrorBoundary>
  );
}

export default App;
