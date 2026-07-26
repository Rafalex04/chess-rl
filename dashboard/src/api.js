const BASE = '/api/runs'

async function getJson(url) {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: ${res.status}`)
  return res.json()
}

export function listRuns() {
  return getJson(BASE)
}

export function getMetrics(runId) {
  return getJson(`${BASE}/${encodeURIComponent(runId)}/metrics`)
}

export function getAccuracy(runId) {
  return getJson(`${BASE}/${encodeURIComponent(runId)}/accuracy`)
}

export function getCheckpoints(runId) {
  return getJson(`${BASE}/${encodeURIComponent(runId)}/checkpoints`)
}

export async function getGamePgn(runId, step, file) {
  const url = `${BASE}/${encodeURIComponent(runId)}/games/${step}/${encodeURIComponent(file)}`
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: ${res.status}`)
  return res.text()
}
