import { useState, useEffect, useCallback } from 'react';
import { api, type Image } from '../api/client';
import ImageDisplay from './ImageDisplay';
import './ImageGallery.css';

interface ImageGalleryProps {
  sessionId: number;
  onRefreshRef?: React.MutableRefObject<() => void>;
}

export default function ImageGallery({ sessionId, onRefreshRef }: ImageGalleryProps) {
  const [images, setImages] = useState<Image[]>([]);
  const [selectedImage, setSelectedImage] = useState<Image | null>(null);
  const [loading, setLoading] = useState(false);

  const loadImages = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.listSessionImages(sessionId);

      // Safety check for data
      if (!data || !Array.isArray(data.items)) {
        console.error('Invalid data received from API:', data);
        setImages([]);
        setSelectedImage(null);
        return;
      }

      setImages(data.items);

      // Use functional update to access current selectedImage without causing dependency issues
      setSelectedImage(currentSelected => {
        // Auto-select first image if:
        // 1. There are images in this session
        // 2. AND (no image is selected OR selected image doesn't belong to this session)
        const selectedImageBelongsToSession = currentSelected &&
          data.items.some(img => img.id === currentSelected.id) &&
          currentSelected.session_id === sessionId;

        if (data.items.length > 0 && !selectedImageBelongsToSession) {
          console.log(`Switching to first image of session ${sessionId}:`, data.items[0]?.id || 'unknown');
          return data.items[0];
        } else if (data.items.length === 0) {
          // Clear selection if session has no images
          console.log(`Session ${sessionId} has no images, clearing selection`);
          return null;
        } else {
          // Keep current selection if it belongs to session
          console.log(`Keeping current selection for session ${sessionId}`);
          return currentSelected;
        }
      });
    } catch (error) {
      console.error('Failed to load images:', error);
      setImages([]);
      setSelectedImage(null);
    } finally {
      setLoading(false);
    }
  }, [sessionId]); // Only sessionId as dependency - prevents infinite loop

  // Reset selected image when session changes
  useEffect(() => {
    console.log(`Session changed to ${sessionId}, resetting selected image`);
    setSelectedImage(null);
    loadImages();
  }, [sessionId, loadImages]);

  useEffect(() => {
    if (onRefreshRef) {
      onRefreshRef.current = loadImages;
    }
  }, [onRefreshRef, loadImages]);

  const handleAnalysisComplete = () => {
    // Refresh images to get updated counts
    loadImages();
  };

  if (loading && images.length === 0) {
    return <div className="loading">Loading images...</div>;
  }

  return (
    <div className="image-gallery">
      <div className="gallery-sidebar">
        <h3>Uploaded Images ({images.length})</h3>
        {images.length === 0 ? (
          <p className="empty-state">No images uploaded yet</p>
        ) : (
          <div className="image-thumbnails">
            {images.map((img) => (
              <div
                key={img.id}
                className={`thumbnail ${selectedImage?.id === img.id ? 'active' : ''}`}
                onClick={() => setSelectedImage(img)}
              >
                <img
                  src={`${api.getImageUrl(img.id)}?t=${img.created_at}`}
                  alt={img.filename}
                  onError={(e) => {
                    // Retry with new timestamp on error
                    const target = e.target as HTMLImageElement;
                    if (!target.dataset.retried) {
                      target.dataset.retried = 'true';
                      setTimeout(() => {
                        target.src = `${api.getImageUrl(img.id)}?t=${Date.now()}`;
                      }, 500);
                    }
                  }}
                />
                <div className="thumbnail-overlay">
                  <span className="thumbnail-filename">{img.filename}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="gallery-main">
        {selectedImage ? (
          <ImageDisplay
            sessionId={sessionId}
            image={selectedImage}
            onAnalysisComplete={handleAnalysisComplete}
          />
        ) : (
          <div className="no-selection">
            <p>Select an image to view and analyze</p>
          </div>
        )}
      </div>
    </div>
  );
}

export { ImageGallery };

