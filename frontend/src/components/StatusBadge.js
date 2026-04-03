export default function StatusBadge({ status }) {
  const map = {
    done:  { color: '#4ade80', bg: 'rgba(22,163,74,0.1)', label: 'Done' },
    error: { color: '#f87171', bg: 'rgba(220,38,38,0.1)', label: 'Error' },
  }
  const s = map[status] || { color: '#f59e0b', bg: 'rgba(245,158,11,0.1)', label: status }
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 6,
      fontSize: 11, fontWeight: 600, color: s.color, background: s.bg,
    }}>
      {s.label}
    </span>
  )
}
