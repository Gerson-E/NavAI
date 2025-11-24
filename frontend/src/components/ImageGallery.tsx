import { useState, useEffect } from 'react';
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

  const loadImages = async () => {
    setLoading(true);
    try {
      const data = await api.listSessionImages(sessionId);
      setImages(data.items);
      if (data.items.length > 0 && !selectedImage) {
        setSelectedImage(data.items[0]);
      }
    } catch (error) {
      console.error('Failed to load images:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadImages();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  useEffect(() => {
    if (onRefreshRef) {
      onRefreshRef.current = loadImages;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onRefreshRef]);

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
                <img src={api.getImageUrl(img.id)} alt={img.filename} />
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

