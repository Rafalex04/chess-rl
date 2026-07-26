import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

export default function AccuracyChart({ accuracy, cursorStep }) {
  if (accuracy.length === 0) {
    return (
      <p className="empty-state">
        Accuracy unavailable for this run (no Stockfish eval was run -- see rlchess/eval.py).
      </p>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={180}>
      <LineChart data={accuracy} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="step" />
        <YAxis label={{ value: 'CP loss', angle: -90, position: 'insideLeft' }} />
        <Tooltip labelFormatter={(s) => `step ${s}`} />
        <Line type="monotone" dataKey="rolling_accuracy_cp" name="Rolling CP loss" stroke="#55A868" dot={false} isAnimationActive={false} />
        {cursorStep != null && <ReferenceLine x={cursorStep} stroke="#333" strokeDasharray="4 4" />}
      </LineChart>
    </ResponsiveContainer>
  )
}
