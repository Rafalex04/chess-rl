export default function RunSelector({ runs, selectedRun, onSelect }) {
  if (runs.length === 0) {
    return <p className="empty-state">No runs found under runs/. Start a training run first.</p>
  }

  return (
    <label className="run-selector">
      Run:{' '}
      <select value={selectedRun ?? ''} onChange={(e) => onSelect(e.target.value)}>
        {runs.map((id) => (
          <option key={id} value={id}>
            {id}
          </option>
        ))}
      </select>
    </label>
  )
}
