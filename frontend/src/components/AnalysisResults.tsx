import React, { useState } from 'react'

type Props = {
  sessionId: string
  imageId: number | null
  imageUrl: string | null
  classifyResult: any | null
  positionResult: any | null
  onRunClassify: () => void
  onRunCheckPosition: (ref: string) => void
  loading: { classify?: boolean; position?: boolean }
}

function colorForClassify(res: any) {
  if (!res) return ''
  if (res.is_kidney === true) return 'green'
  if (res.detected_organ === 'unknown') return 'yellow'
  return 'red'
}

function colorForVerdict(v?: string) {
  if (v === 'good') return 'green'
  if (v === 'borderline') return 'yellow'
  if (v === 'poor') return 'red'
  return ''
}

export default function AnalysisResults(props: Props) {
  const [refView, setRefView] = useState('kidney_long_axis')
  const { sessionId, imageId, imageUrl, classifyResult, positionResult, onRunClassify, onRunCheckPosition, loading } = props

  return (
    <div className="panel">
      <h2>Analysis</h2>

      <div>
        <strong>Session:</strong> {sessionId || '-'} <br />
        <strong>Image ID:</strong> {imageId ?? '-'}
      </div>

      {imageUrl && (
        <div className="thumb small">
          <img src={imageUrl} alt="uploaded" />
        </div>
      )}

      <div className="section">
        <h3>Kidney detection</h3>
        <button onClick={onRunClassify} disabled={!imageId || !!loading.classify}>
          {loading.classify ? 'Running...' : 'Run kidney detection'}
        </button>

        {classifyResult && (
          <div className="result" data-color={colorForClassify(classifyResult)}>
            <div><strong>Detected organ:</strong> {String(classifyResult.detected_organ ?? '-')}</div>
            <div><strong>Is kidney:</strong> {String(classifyResult.is_kidney ?? '-')}</div>
            <div><strong>Confidence:</strong> {String(classifyResult.confidence ?? '-')}</div>
            <div><strong>Message:</strong> {String(classifyResult.message ?? '-')}</div>
          </div>
        )}
      </div>

      <div className="section">
        <h3>Probe position check</h3>
        <label>
          Reference view id
          <input value={refView} onChange={e => setRefView(e.target.value)} />
        </label>
        <button onClick={() => onRunCheckPosition(refView)} disabled={!imageId || !!loading.position}>
          {loading.position ? 'Running...' : 'Run position check'}
        </button>

        {positionResult && (
          <div className="result" data-color={colorForVerdict(positionResult.verdict)}>
            <div><strong>Verdict:</strong> {String(positionResult.verdict ?? '-')}</div>
            <div><strong>SSIM:</strong> {String(positionResult.ssim ?? '-')}</div>
            <div><strong>NCC:</strong> {String(positionResult.ncc ?? '-')}</div>
            <div><strong>Confidence:</strong> {String(positionResult.confidence ?? '-')}</div>
            <div><strong>Message:</strong> {String(positionResult.message ?? '-')}</div>
          </div>
        )}
      </div>
    </div>
  )
}
