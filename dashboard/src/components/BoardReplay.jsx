import { useEffect, useRef, useState } from 'react'
import { Chess } from 'chess.js'
import { Chessboard } from 'react-chessboard'
import { getGamePgn } from '../api.js'

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

export default function BoardReplay({ runId, step, files }) {
  const [selectedFile, setSelectedFile] = useState(files[0] ?? null)
  const [moves, setMoves] = useState([])
  const [result, setResult] = useState(null)
  const [moveIndex, setMoveIndex] = useState(0)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    setSelectedFile(files[0] ?? null)
  }, [files])

  useEffect(() => {
    if (!selectedFile) {
      setMoves([])
      return
    }
    let cancelled = false
    getGamePgn(runId, step, selectedFile).then((pgn) => {
      if (cancelled) return
      const chess = new Chess()
      chess.loadPgn(pgn)
      setMoves(chess.history({ verbose: true }))
      setResult(chess.header().Result ?? null)
      setMoveIndex(0)
    })
    return () => {
      cancelled = true
    }
  }, [runId, step, selectedFile])

  const intervalRef = useRef(null)
  useEffect(() => {
    if (!playing) return
    intervalRef.current = setInterval(() => {
      setMoveIndex((i) => {
        if (i >= moves.length) {
          setPlaying(false)
          return i
        }
        return i + 1
      })
    }, 500)
    return () => clearInterval(intervalRef.current)
  }, [playing, moves.length])

  if (!selectedFile) {
    return <p className="empty-state">No sampled games at this checkpoint.</p>
  }

  const fen = moveIndex === 0 ? START_FEN : moves[moveIndex - 1]?.after ?? START_FEN

  return (
    <div className="board-replay">
      <div className="board-replay-controls">
        {files.length > 1 && (
          <select value={selectedFile} onChange={(e) => setSelectedFile(e.target.value)}>
            {files.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        )}
        <span>
          move {moveIndex} / {moves.length}
          {result ? ` (${result})` : ''}
        </span>
      </div>

      <Chessboard position={fen} arePiecesDraggable={false} boardWidth={360} />

      <div className="board-replay-controls">
        <button onClick={() => setMoveIndex(0)} disabled={moveIndex === 0}>
          |&lt;
        </button>
        <button onClick={() => setMoveIndex((i) => Math.max(0, i - 1))} disabled={moveIndex === 0}>
          &lt;
        </button>
        <button onClick={() => setPlaying((p) => !p)}>{playing ? 'Pause' : 'Play'}</button>
        <button
          onClick={() => setMoveIndex((i) => Math.min(moves.length, i + 1))}
          disabled={moveIndex >= moves.length}
        >
          &gt;
        </button>
        <button onClick={() => setMoveIndex(moves.length)} disabled={moveIndex >= moves.length}>
          &gt;|
        </button>
        <input
          type="range"
          min={0}
          max={moves.length}
          value={moveIndex}
          onChange={(e) => setMoveIndex(Number(e.target.value))}
        />
      </div>
    </div>
  )
}
