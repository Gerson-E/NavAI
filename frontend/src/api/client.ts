import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface Session {
  id: number;
  user_id: number;
  patient_identifier: string | null;
  status: 'active' | 'completed' | 'cancelled';
  notes: string | null;
  created_at: string;
  updated_at: string;
  image_count?: number | null;
  comparison_count?: number | null;
}

export interface SessionCreate {
  patient_identifier?: string | null;
  notes?: string | null;
}

export interface Image {
  id: number;
  session_id: number;
  filename: string;
  file_size: number;
  mime_type: string;
  width: number | null;
  height: number | null;
  url: string;
  thumbnail_url: string | null;
  created_at: string;
  dimensions: string | null;
  file_size_formatted: string | null;
}

export interface ComparisonResponse {
  id: number;
  session_id: number;
  image_id: number;
  reference_view_id: string;
  ssim: number;
  ncc: number;
  verdict: 'good' | 'borderline' | 'poor';
  message: string;
  confidence: number;
  processing_time_ms: number | null;
  created_at: string;
}

export interface ClassificationResponse {
  id: number;
  session_id: number;
  image_id: number;
  detected_organ: string;
  confidence: number;
  is_kidney: boolean;
  message: string;
  processing_time_ms: number | null;
  created_at: string;
}

// API functions
export const api = {
  // Sessions
  createSession: async (data: SessionCreate): Promise<Session> => {
    const response = await apiClient.post('/api/v1/sessions', data);
    return response.data;
  },

  getSession: async (id: number): Promise<Session> => {
    const response = await apiClient.get(`/api/v1/sessions/${id}`);
    return response.data;
  },

  listSessions: async (): Promise<{ items: Session[]; total: number }> => {
    const response = await apiClient.get('/api/v1/sessions');
    return response.data;
  },

  deleteSession: async (sessionId: number): Promise<void> => {
    await apiClient.delete(`/api/v1/sessions/${sessionId}`);
  },

  // Images
  uploadImage: async (sessionId: number, file: File): Promise<Image> => {
    const formData = new FormData();
    formData.append('file', file);
    // Use a separate axios instance without default JSON headers for file uploads
    const response = await axios.post(
      `${API_BASE_URL}/api/v1/sessions/${sessionId}/images`,
      formData,
      {
        // Don't set Content-Type - axios will set it automatically with boundary
        headers: {},
      }
    );
    return response.data;
  },

  getImageUrl: (imageId: number): string => {
    return `${API_BASE_URL}/api/v1/images/${imageId}/file`;
  },

  listSessionImages: async (sessionId: number): Promise<{ items: Image[]; total: number }> => {
    const response = await apiClient.get(`/api/v1/sessions/${sessionId}/images`);
    return response.data;
  },

  // Analysis
  checkPosition: async (sessionId: number, imageId: number, referenceViewId: string): Promise<ComparisonResponse> => {
    const response = await apiClient.post('/api/v1/check-position', {
      session_id: sessionId,
      image_id: imageId,
      reference_view_id: referenceViewId,
    });
    return response.data;
  },

  classifyOrgan: async (sessionId: number, imageId: number): Promise<ClassificationResponse> => {
    const response = await apiClient.post('/api/v1/classify-organ', {
      session_id: sessionId,
      image_id: imageId,
    });
    return response.data;
  },

  getSessionComparisons: async (sessionId: number): Promise<{
    items: ComparisonResponse[];
    total: number;
    average_ssim: number | null;
    average_ncc: number | null;
    verdict_distribution: { good: number; borderline: number; poor: number } | null;
  }> => {
    const response = await apiClient.get(`/api/v1/sessions/${sessionId}/comparisons`);
    return response.data;
  },
};

export default apiClient;

