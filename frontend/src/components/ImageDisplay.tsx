import { useState, useEffect } from 'react';
import { api, type Image, type ComparisonResponse, type ClassificationResponse } from '../api/client';
import './ImageDisplay.css';

interface ImageDisplayProps {
  sessionId: number;
  image: Image;
  onAnalysisComplete?: () => void;
}

export default function ImageDisplay({ sessionId, image, onAnalysisComplete }: ImageDisplayProps) {
  // ALL HOOKS MUST BE AT THE TOP - React Rules of Hooks
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [classification, setClassification] = useState<ClassificationResponse | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzingType, setAnalyzingType] = useState<'position' | 'organ' | null>(null);
  const [referenceView, setReferenceView] = useState('kidney_longitudinal');
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);
  const [imageKey, setImageKey] = useState(Date.now());

  // Reset image loading state when image changes
  useEffect(() => {
    setImageLoaded(false);
    setImageError(false);
    setImageKey(Date.now()); // Force reload with new timestamp
  }, [image?.id]);

  // Safety check AFTER all hooks: Only block if image is missing
  if (!image) {
    console.log('ImageDisplay: No image provided, showing placeholder');
    return (
      <div className="image-display">
        <div className="no-selection" style={{
          minHeight: '400px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'var(--bg-color)',
          borderRadius: 'var(--radius-lg)',
          color: 'var(--text-secondary)'
        }}>
          <p>Loading session images...</p>
        </div>
      </div>
    );
  }

  // Warn if session mismatch but still render (less strict)
  if (image.session_id !== sessionId) {
    console.warn(`Session mismatch: Image ${image.id} belongs to session ${image.session_id}, but current session is ${sessionId}. Rendering anyway.`);
  }

  const analyzePosition = async () => {
    setAnalyzing(true);
    setAnalyzingType('position');
    try {
      console.log('Analyzing position with:', {
        sessionId,
        imageId: image.id,
        referenceView,
        imageFilename: image.filename
      });
      
      const result = await api.checkPosition(sessionId, image.id, referenceView);
      console.log('Analysis result:', result);
      setComparison(result);
      onAnalysisComplete?.();
    } catch (error: any) {
      console.error('Position analysis failed:', error);
      console.error('Error response:', error.response);
      console.error('Error data:', error.response?.data);
      console.error('Full error object:', error);
      
      let errorMessage = 'Failed to analyze position.\n\n';
      if (error.response?.data?.detail) {
        errorMessage += `Backend error: ${error.response.data.detail}`;
      } else if (error.response?.status) {
        errorMessage += `HTTP ${error.response.status}: ${error.response.statusText || 'Unknown error'}`;
      } else if (error.message) {
        errorMessage += `Error: ${error.message}`;
      } else {
        errorMessage += 'Please check the browser console and backend logs for details.';
      }
      
      errorMessage += `\n\nRequest details:\n- Session ID: ${sessionId}\n- Image ID: ${image.id}\n- Reference View: ${referenceView}`;
      
      alert(errorMessage);
    } finally {
      setAnalyzing(false);
      setAnalyzingType(null);
    }
  };

  const analyzeOrgan = async () => {
    setAnalyzing(true);
    setAnalyzingType('organ');
    try {
      const result = await api.classifyOrgan(sessionId, image.id);
      setClassification(result);
      onAnalysisComplete?.();
    } catch (error: any) {
      console.error('Organ classification failed:', error);
      console.error('Full error:', JSON.stringify(error, null, 2));
      
      let errorMessage = 'Failed to classify organ. ';
      if (error.response?.data?.detail) {
        errorMessage += error.response.data.detail;
      } else if (error.message) {
        errorMessage += error.message;
      } else {
        errorMessage += 'Please check the backend logs for details.';
      }
      
      alert(errorMessage);
    } finally {
      setAnalyzing(false);
      setAnalyzingType(null);
    }
  };

  const getVerdictColor = (verdict: string) => {
    switch (verdict) {
      case 'good':
        return '#10b981'; // green
      case 'borderline':
        return '#f59e0b'; // yellow
      case 'poor':
        return '#ef4444'; // red
      default:
        return '#6b7280'; // gray
    }
  };

  const handleImageLoad = () => {
    setImageLoaded(true);
    setImageError(false);
  };

  const handleImageError = () => {
    console.error('Failed to load image:', image.id);
    setImageError(true);
    setImageLoaded(true);

    // Retry after a short delay
    setTimeout(() => {
      setImageKey(Date.now());
      setImageError(false);
      setImageLoaded(false);
    }, 1000);
  };

  // Add cache-busting timestamp to ensure fresh image load
  const imageUrl = `${api.getImageUrl(image.id)}?t=${imageKey}`;

  return (
    <div className="image-display">
      <div className="image-container">
        {!imageLoaded && !imageError && (
          <div className="image-loading" style={{
            minHeight: '300px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--text-secondary)'
          }}>
            <div>
              <div className="spinner" style={{ margin: '0 auto 1rem' }}></div>
              <p>Loading image...</p>
            </div>
          </div>
        )}
        {imageError && (
          <div className="image-error" style={{
            minHeight: '300px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--error-color)'
          }}>
            <p>Retrying image load...</p>
          </div>
        )}
        <img
          src={imageUrl}
          alt={image.filename}
          className="uploaded-image"
          onLoad={handleImageLoad}
          onError={handleImageError}
          style={{ display: imageLoaded && !imageError ? 'block' : 'none' }}
        />
        {imageLoaded && !imageError && (
          <div className="image-info">
            <p className="image-filename">{image.filename}</p>
            {image.dimensions && (
              <p className="image-meta">{image.dimensions} • {image.file_size_formatted}</p>
            )}
          </div>
        )}
      </div>

      <div className="analysis-controls">
        <div className="analysis-section">
          <h3>Position Analysis</h3>
          <div className="reference-selector">
            <label>Reference View:</label>
            <select
              value={referenceView}
              onChange={(e) => setReferenceView(e.target.value)}
              disabled={analyzing}
            >
              <option value="kidney_longitudinal">Kidney Longitudinal</option>
              <option value="cardiac_4chamber">Cardiac 4-Chamber</option>
              <option value="cardiac_parasternal_long">Cardiac Parasternal Long</option>
              <option value="liver_standard">Liver Standard</option>
            </select>
          </div>
          <button
            className="btn btn-primary"
            onClick={analyzePosition}
            disabled={analyzing}
          >
            {analyzing && analyzingType === 'position' ? 'Analyzing...' : 'Check Position'}
          </button>
        </div>

        <div className="analysis-section">
          <h3>Organ Classification</h3>
          <button
            className="btn btn-secondary"
            onClick={analyzeOrgan}
            disabled={analyzing}
          >
            {analyzing && analyzingType === 'organ' ? 'Classifying...' : 'Classify Organ'}
          </button>
        </div>
      </div>

      {comparison && (
        <div className="analysis-result position-result">
          <div className="result-header">
            <h3>Position Analysis Result</h3>
            <span
              className="verdict-badge"
              style={{ backgroundColor: getVerdictColor(comparison.verdict) }}
            >
              {comparison.verdict.toUpperCase()}
            </span>
          </div>
          <div className="result-scores">
            <div className="score-item">
              <span className="score-label">SSIM:</span>
              <span className="score-value">{comparison.ssim.toFixed(3)}</span>
            </div>
            <div className="score-item">
              <span className="score-label">NCC:</span>
              <span className="score-value">{comparison.ncc.toFixed(3)}</span>
            </div>
            <div className="score-item">
              <span className="score-label">Confidence:</span>
              <span className="score-value">{(comparison.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
          <div className="result-message">
            <p>{comparison.message}</p>
          </div>
          {comparison.processing_time_ms && (
            <div className="result-meta">
              Processed in {comparison.processing_time_ms}ms
            </div>
          )}
        </div>
      )}

      {classification && (
        <div className="analysis-result classification-result">
          <div className="result-header">
            <h3>Organ Classification Result</h3>
            {classification.is_kidney && (
              <span className="kidney-badge">KIDNEY DETECTED</span>
            )}
          </div>
          <div className="result-scores">
            <div className="score-item">
              <span className="score-label">Detected Organ:</span>
              <span className="score-value">{classification.detected_organ}</span>
            </div>
            <div className="score-item">
              <span className="score-label">Confidence:</span>
              <span className="score-value">{(classification.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
          <div className="result-message">
            <p>{classification.message}</p>
          </div>
          {classification.processing_time_ms && (
            <div className="result-meta">
              Processed in {classification.processing_time_ms}ms
            </div>
          )}
        </div>
      )}
    </div>
  );
}

