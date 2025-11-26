import { useState, useEffect } from 'react';
import { api, type Session, type SessionCreate } from '../api/client';
import './SessionManager.css';

interface SessionManagerProps {
  currentSession: Session | null;
  onSessionChange: (session: Session | null) => void;
}

export default function SessionManager({ currentSession, onSessionChange }: SessionManagerProps) {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [patientId, setPatientId] = useState('');
  const [notes, setNotes] = useState('');

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      const data = await api.listSessions();
      setSessions(data.items);
    } catch (error) {
      console.error('Failed to load sessions:', error);
    }
  };

  const createSession = async () => {
    if (loading) return;

    setLoading(true);
    try {
      const data: SessionCreate = {
        patient_identifier: patientId || null,
        notes: notes || null,
      };
      const newSession = await api.createSession(data);
      console.log('Created new session:', newSession.id);

      // Update sessions list
      setSessions([newSession, ...sessions]);

      // Clear form
      setShowCreateForm(false);
      setPatientId('');
      setNotes('');

      // Switch to new session (do this last to avoid race conditions)
      console.log('Switching to new session:', newSession.id);
      onSessionChange(newSession);
    } catch (error) {
      console.error('Failed to create session:', error);
      alert('Failed to create session. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const selectSession = (session: Session) => {
    onSessionChange(session);
  };

  const deleteSession = async (sessionId: number, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent selecting the session when clicking delete
    
    const session = sessions.find(s => s.id === sessionId);
    const sessionLabel = session?.patient_identifier || `Session #${sessionId}`;
    
    if (!confirm(`Are you sure you want to delete ${sessionLabel}?\n\nThis will permanently delete:\n- All images in this session\n- All analysis results\n- The session itself\n\nThis action cannot be undone.`)) {
      return;
    }

    setLoading(true);
    try {
      await api.deleteSession(sessionId);
      
      // Remove from list
      const updatedSessions = sessions.filter(s => s.id !== sessionId);
      setSessions(updatedSessions);
      
      // If deleted session was current, clear it
      if (currentSession?.id === sessionId) {
        onSessionChange(null);
      }
    } catch (error: any) {
      console.error('Failed to delete session:', error);
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to delete session. Please try again.';
      alert(`Error: ${errorMessage}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="session-manager">
      <div className="session-manager-header">
        <h2>Sessions</h2>
        <button
          className="btn btn-primary"
          onClick={() => setShowCreateForm(!showCreateForm)}
        >
          {showCreateForm ? 'Cancel' : '+ New Session'}
        </button>
      </div>

      {showCreateForm && (
        <div className="create-session-form">
          <input
            type="text"
            placeholder="Patient ID (optional)"
            value={patientId}
            onChange={(e) => setPatientId(e.target.value)}
            className="input"
          />
          <textarea
            placeholder="Notes (optional)"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            className="input"
            rows={3}
          />
          <button
            className="btn btn-primary"
            onClick={createSession}
            disabled={loading}
          >
            {loading ? 'Creating...' : 'Create Session'}
          </button>
        </div>
      )}

      <div className="session-list">
        {sessions.length === 0 ? (
          <p className="empty-state">No sessions yet. Create one to get started!</p>
        ) : (
          sessions.map((session) => (
            <div
              key={session.id}
              className={`session-item ${currentSession?.id === session.id ? 'active' : ''}`}
              onClick={() => selectSession(session)}
            >
              <div className="session-item-header">
                <span className="session-id">Session #{session.id}</span>
                <div className="session-header-right">
                  <span className={`session-status status-${session.status}`}>
                    {session.status}
                  </span>
                  <button
                    className="btn-delete-session"
                    onClick={(e) => deleteSession(session.id, e)}
                    disabled={loading}
                    title="Delete session"
                  >
                    ×
                  </button>
                </div>
              </div>
              {session.patient_identifier && (
                <div className="session-patient">Patient: {session.patient_identifier}</div>
              )}
              <div className="session-stats">
                <span>{session.image_count || 0} images</span>
                <span>•</span>
                <span>{session.comparison_count || 0} analyses</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

