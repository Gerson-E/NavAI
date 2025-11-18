import React, { useState } from 'react'
import { uploadImage, resolveImageUrl } from '../api'

type Props = {
  sessionId: string
  setSessionId: (s: string) => void
  onUploaded: (data: any, previewUrl: string | null) => void
  setError: (e: string | null) => void
  setImageUrl: (u: string | null) => void
}

export default function UploadPanel({ sessionId, setSessionId, onUploaded, setError, setImageUrl }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null
    setFile(f)
    setError(null)
    if (f) {
      const url = URL.createObjectURL(f)
      setPreview(url)
      setImageUrl(url)
    } else {
      setPreview(null)
      setImageUrl(null)
    }
  }

  const doUpload = async () => {
    setError(null)
    if (!file) {
      setError('No file selected')
      return
    }
    if (!sessionId || isNaN(Number(sessionId))) {
      setError('Enter a numeric session_id')
      return
    }
    setLoading(true)
    try {
      const data = await uploadImage(Number(sessionId), file)
      const serverImageUrl = resolveImageUrl(data?.url ?? data?.storage_path)
      onUploaded(data, serverImageUrl ?? preview)
    } catch (err: any) {
      setError(err.message || String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="panel">
      <h2>Upload Image</h2>
      <label>
        Session ID
        <input
          value={sessionId}
          onChange={e => setSessionId(e.target.value)}
          placeholder="e.g. 123"
        />
      </label>

      <label>
        Select image
        <input type="file" accept="image/*" onChange={onFileChange} />
      </label>

      {preview && (
        <div className="thumb">
          <img src={preview} alt="preview" />
        </div>
      )}

      <div className="actions">
        <button onClick={doUpload} disabled={loading}>
          {loading ? 'Uploading...' : 'Upload'}
        </button>
      </div>
    </div>
  )
}
