import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

export default function WinChart({ metrics, cursorStep }) {
  if (metrics.length === 0) {
    return <p className="empty-state">No metrics yet.</p>
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={metrics} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="step" />
        <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
        <Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} labelFormatter={(s) => `step ${s}`} />
        <Legend />
        <Line type="monotone" dataKey="rolling_white_winrate" name="White" stroke="#4C72B0" dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="rolling_black_winrate" name="Black" stroke="#C44E52" dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="rolling_draw_rate" name="Draw" stroke="#8C8C8C" dot={false} isAnimationActive={false} />
        {cursorStep != null && <ReferenceLine x={cursorStep} stroke="#333" strokeDasharray="4 4" />}
      </LineChart>
    </ResponsiveContainer>
  )
}
