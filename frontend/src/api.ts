const API_BASE = (import.meta.env.VITE_API_BASE as string) || 'http://localhost:8000'

function handleResp(res: Response) {
  if (res.ok) return res.json().catch(() => ({}))
  return res.text().then(text => {
    throw new Error(text || `HTTP ${res.status}`)
  })
}

export async function uploadImage(sessionId: number, file: File) {
  const url = `${API_BASE}/api/v1/sessions/${sessionId}/images`
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch(url, { method: 'POST', body: fd })
  return handleResp(res)
}

export async function classifyOrgan(sessionId: number, imageId: number) {
  const url = `${API_BASE}/api/v1/classify-organ`
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, image_id: imageId })
  })
  return handleResp(res)
}

export async function checkPosition(sessionId: number, imageId: number, referenceViewId: string) {
  const url = `${API_BASE}/api/v1/check-position`
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, image_id: imageId, reference_view_id: referenceViewId })
  })
  return handleResp(res)
}

export function resolveImageUrl(storagePath: string | undefined) {
  if (!storagePath) return null
  if (storagePath.startsWith('http') || storagePath.startsWith('/')) {
    const cleaned = storagePath.startsWith('/') ? storagePath.slice(1) : storagePath
    return `${API_BASE}/${cleaned}`
  }
  return `${API_BASE}/${storagePath}`
}
