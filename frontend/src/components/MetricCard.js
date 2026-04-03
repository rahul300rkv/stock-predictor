export default function MetricCard({ label, value, sub, accent }) {
  return (
    <div style={{
      background: '#0f172a',
      border: `1px solid ${accent ? 'rgba(22,163,74,0.4)' : '#1e293b'}`,
      borderRadius: 12,
      padding: '16px 20px',
    }}>
      <div style={{ fontSize: 11, color: '#475569', letterSpacing: '0.08em', marginBottom: 8 }}>
        {label.toUpperCase()}
      </div>
      <div style={{
        fontSize: 24, fontWeight: 700, letterSpacing: '-0.02em',
        color: accent ? '#4ade80' : '#e2e8f0',
      }}>
        {value}
      </div>
      <div style={{ fontSize: 11, color: '#475569', marginTop: 4 }}>{sub}</div>
    </div>
  )
}
