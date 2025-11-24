import { useState } from 'react';
import { api, type Image, type ComparisonResponse, type ClassificationResponse } from '../api/client';
import './ImageDisplay.css';

interface ImageDisplayProps {
  sessionId: number;
  image: Image;
  onAnalysisComplete?: () => void;
}

export default function ImageDisplay({ sessionId, image, onAnalysisComplete }: ImageDisplayProps) {
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [classification, setClassification] = useState<ClassificationResponse | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzingType, setAnalyzingType] = useState<'position' | 'organ' | null>(null);
  const [referenceView, setReferenceView] = useState('kidney_longitudinal');

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

  const imageUrl = api.getImageUrl(image.id);

  return (
    <div className="image-display">
      <div className="image-container">
        <img src={imageUrl} alt={image.filename} className="uploaded-image" />
        <div className="image-info">
          <p className="image-filename">{image.filename}</p>
          {image.dimensions && (
            <p className="image-meta">{image.dimensions} • {image.file_size_formatted}</p>
          )}
        </div>
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

