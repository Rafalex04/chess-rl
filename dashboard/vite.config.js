import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const runsDir = path.resolve(__dirname, '../runs')

function readJsonl(filePath) {
  if (!fs.existsSync(filePath)) return []
  return fs
    .readFileSync(filePath, 'utf-8')
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line))
}

function sendJson(res, data) {
  res.setHeader('Content-Type', 'application/json')
  res.end(JSON.stringify(data))
}

function notFound(res) {
  res.statusCode = 404
  res.end('not found')
}

// Serves runs/<run-id>/ as a tiny read-only JSON API for the dashboard, so
// `npm run dev` alone is enough (no separate backend process) -- matches
// planning/SPEC.md sec 7 ("served statically or via a tiny endpoint").
function runsApiPlugin() {
  return {
    name: 'runs-api',
    configureServer(server) {
      server.middlewares.use('/api/runs', (req, res) => {
        try {
          handleRequest(req, res)
        } catch (err) {
          res.statusCode = 500
          res.end(String(err))
        }
      })
    },
  }
}

function handleRequest(req, res) {
  // Connect strips the '/api/runs' mount prefix, so req.url is relative to it.
  const { pathname } = new URL(req.url, 'http://localhost')
  const parts = pathname.split('/').filter(Boolean)

  if (parts.length === 0) {
    const ids = fs.existsSync(runsDir)
      ? fs.readdirSync(runsDir).filter((name) => fs.statSync(path.join(runsDir, name)).isDirectory())
      : []
    sendJson(res, ids.sort())
    return
  }

  const runId = decodeURIComponent(parts[0])
  const runDir = path.join(runsDir, runId)
  const resource = parts[1]

  if (resource === 'metrics') {
    sendJson(res, readJsonl(path.join(runDir, 'metrics.jsonl')))
    return
  }

  if (resource === 'accuracy') {
    sendJson(res, readJsonl(path.join(runDir, 'accuracy.jsonl')))
    return
  }

  if (resource === 'checkpoints') {
    const gamesDir = path.join(runDir, 'games')
    const checkpoints = fs.existsSync(gamesDir)
      ? fs
          .readdirSync(gamesDir)
          .filter((name) => name.startsWith('ckpt_'))
          .map((name) => ({
            step: parseInt(name.slice('ckpt_'.length), 10),
            files: fs
              .readdirSync(path.join(gamesDir, name))
              .filter((f) => f.endsWith('.pgn'))
              .sort(),
          }))
          .sort((a, b) => a.step - b.step)
      : []
    sendJson(res, checkpoints)
    return
  }

  if (resource === 'games' && parts.length === 4) {
    const step = parts[2]
    const file = decodeURIComponent(parts[3])
    const pgnPath = path.join(runDir, 'games', `ckpt_${step}`, file)
    if (!fs.existsSync(pgnPath)) {
      notFound(res)
      return
    }
    res.setHeader('Content-Type', 'text/plain')
    res.end(fs.readFileSync(pgnPath, 'utf-8'))
    return
  }

  notFound(res)
}

export default defineConfig({
  plugins: [react(), runsApiPlugin()],
})
