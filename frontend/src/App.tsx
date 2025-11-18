import React, { useState } from 'react'
import UploadPanel from './components/UploadPanel'
import AnalysisResults from './components/AnalysisResults'
import { classifyOrgan, checkPosition } from './api'

export default function App() {
  const [sessionId, setSessionId] = useState<string>('')
  const [imageId, setImageId] = useState<number | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [classifyResult, setClassifyResult] = useState<any | null>(null)
  const [positionResult, setPositionResult] = useState<any | null>(null)
  const [loading, setLoading] = useState<{ classify?: boolean; position?: boolean }>({})

  const handleClassify = async () => {
    setError(null)
    if (!sessionId || !imageId) {
      setError('Missing session_id or image_id')
      return
    }
    setLoading(s => ({ ...s, classify: true }))
    try {
      const res = await classifyOrgan(Number(sessionId), imageId)
      setClassifyResult(res)
    } catch (err: any) {
      setError(err.message || String(err))
    } finally {
      setLoading(s => ({ ...s, classify: false }))
    }
  }

  const handleCheckPosition = async (referenceViewId: string) => {
    setError(null)
    if (!sessionId || !imageId) {
      setError('Missing session_id or image_id')
      return
    }
    setLoading(s => ({ ...s, position: true }))
    try {
      const res = await checkPosition(Number(sessionId), imageId, referenceViewId)
      setPositionResult(res)
    } catch (err: any) {
      setError(err.message || String(err))
    } finally {
      setLoading(s => ({ ...s, position: false }))
    }
  }

  return (
    <div className="app">
      <h1>NavAI — Ultrasound Frontend</h1>

      <div className="layout">
        <UploadPanel
          sessionId={sessionId}
          setSessionId={setSessionId}
          onUploaded={(data, previewUrl) => {
            setImageId(data?.id ?? null)
            setImageUrl(previewUrl ?? null)
            setClassifyResult(null)
            setPositionResult(null)
            setError(null)
          }}
          setError={setError}
          setImageUrl={setImageUrl}
        />

        <div className="right">
          {error && <div className="alert">{error}</div>}

          <AnalysisResults
            sessionId={sessionId}
            imageId={imageId}
            imageUrl={imageUrl}
            classifyResult={classifyResult}
            positionResult={positionResult}
            onRunClassify={handleClassify}
            onRunCheckPosition={handleCheckPosition}
            loading={loading}
          />
        </div>
      </div>

      <footer>
        <small>Frontend assumes backend at VITE_API_BASE (see frontend/.env.example)</small>
      </footer>
    </div>
  )
}
