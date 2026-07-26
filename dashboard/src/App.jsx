import { useEffect, useRef, useState } from 'react'
import { getAccuracy, getCheckpoints, getMetrics, listRuns } from './api.js'
import RunSelector from './components/RunSelector.jsx'
import WinChart from './components/WinChart.jsx'
import AccuracyChart from './components/AccuracyChart.jsx'
import BoardReplay from './components/BoardReplay.jsx'

export default function App() {
  const [runs, setRuns] = useState([])
  const [selectedRun, setSelectedRun] = useState(null)
  const [metrics, setMetrics] = useState([])
  const [accuracy, setAccuracy] = useState([])
  const [checkpoints, setCheckpoints] = useState([])
  const [checkpointIdx, setCheckpointIdx] = useState(0)
  const [timelapsePlaying, setTimelapsePlaying] = useState(false)

  useEffect(() => {
    listRuns().then((ids) => {
      setRuns(ids)
      if (ids.length > 0) setSelectedRun(ids[ids.length - 1])
    })
  }, [])

  useEffect(() => {
    if (!selectedRun) return
    getMetrics(selectedRun).then(setMetrics)
    getAccuracy(selectedRun).then(setAccuracy)
    getCheckpoints(selectedRun).then((ckpts) => {
      setCheckpoints(ckpts)
      setCheckpointIdx(Math.max(0, ckpts.length - 1))
    })
  }, [selectedRun])

  const intervalRef = useRef(null)
  useEffect(() => {
    if (!timelapsePlaying) return
    intervalRef.current = setInterval(() => {
      setCheckpointIdx((i) => {
        if (i >= checkpoints.length - 1) {
          setTimelapsePlaying(false)
          return i
        }
        return i + 1
      })
    }, 2000)
    return () => clearInterval(intervalRef.current)
  }, [timelapsePlaying, checkpoints.length])

  const currentCheckpoint = checkpoints[checkpointIdx]

  return (
    <div className="app">
      <header>
        <h1>RL Chess Self-Play</h1>
        <RunSelector runs={runs} selectedRun={selectedRun} onSelect={setSelectedRun} />
      </header>

      {selectedRun && (
        <>
          <section className="charts">
            <div className="chart-panel">
              <h2>Win rate (rolling)</h2>
              <WinChart metrics={metrics} cursorStep={currentCheckpoint?.step} />
            </div>
            <div className="chart-panel">
              <h2>Accuracy (rolling centipawn loss)</h2>
              <AccuracyChart accuracy={accuracy} cursorStep={currentCheckpoint?.step} />
            </div>
          </section>

          <section className="timelapse">
            <h2>Timelapse</h2>
            {checkpoints.length === 0 ? (
              <p className="empty-state">No checkpoints with sampled games yet.</p>
            ) : (
              <>
                <div className="board-replay-controls">
                  <button onClick={() => setTimelapsePlaying((p) => !p)}>
                    {timelapsePlaying ? 'Pause' : 'Play'} timelapse
                  </button>
                  <span>
                    checkpoint step {currentCheckpoint?.step} ({checkpointIdx + 1}/{checkpoints.length})
                  </span>
                  <input
                    type="range"
                    min={0}
                    max={checkpoints.length - 1}
                    value={checkpointIdx}
                    onChange={(e) => setCheckpointIdx(Number(e.target.value))}
                  />
                </div>
                <BoardReplay
                  key={currentCheckpoint.step}
                  runId={selectedRun}
                  step={currentCheckpoint.step}
                  files={currentCheckpoint.files}
                />
              </>
            )}
          </section>
        </>
      )}
    </div>
  )
}
