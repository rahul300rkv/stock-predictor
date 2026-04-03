export default function OHLCVTable({ data }) {
  if (!data?.length) return null
  const rows = [...data].reverse()

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr>
            {['Date','Open','High','Low','Close','Volume'].map(h => (
              <th key={h} style={{
                textAlign: h === 'Date' ? 'left' : 'right',
                padding: '8px 12px',
                fontSize: 11, fontWeight: 500,
                color: '#475569', letterSpacing: '0.06em',
                borderBottom: '1px solid #1e293b',
              }}>{h.toUpperCase()}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const up = row.close >= row.open
            return (
              <tr key={row.date} style={{
                background: i % 2 === 0 ? 'transparent' : 'rgba(15,23,42,0.5)',
              }}>
                <td style={{ padding: '8px 12px', color: '#94a3b8', fontFamily: 'var(--font-mono)' }}>
                  {row.date}
                </td>
                {['open','high','low'].map(k => (
                  <td key={k} style={{ padding: '8px 12px', textAlign: 'right', color: '#64748b', fontFamily: 'var(--font-mono)' }}>
                    ₹{row[k].toLocaleString('en-IN')}
                  </td>
                ))}
                <td style={{
                  padding: '8px 12px', textAlign: 'right', fontWeight: 600,
                  color: up ? '#4ade80' : '#f87171',
                  fontFamily: 'var(--font-mono)',
                }}>
                  ₹{row.close.toLocaleString('en-IN')}
                </td>
                <td style={{ padding: '8px 12px', textAlign: 'right', color: '#64748b', fontFamily: 'var(--font-mono)' }}>
                  {(row.volume / 1e5).toFixed(1)}L
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
