'use client'
import { useState, useEffect, useRef } from 'react'

export default function StockSearch({ symbols, value, onChange, disabled }) {
  const [query, setQuery]       = useState('')
  const [open, setOpen]         = useState(false)
  const [filtered, setFiltered] = useState([])
  const inputRef = useRef(null)
  const listRef  = useRef(null)

  // Display label for current value
  const currentLabel = symbols.find(s => s.value === value)?.label || value || ''

  useEffect(() => {
    if (!query) {
      setFiltered(symbols.slice(0, 80))
    } else {
      const q = query.toLowerCase()
      setFiltered(
        symbols
          .filter(s => s.label.toLowerCase().includes(q) || s.value.toLowerCase().includes(q))
          .slice(0, 80)
      )
    }
  }, [query, symbols])

  const select = (sym) => {
    onChange(sym.value)
    setQuery('')
    setOpen(false)
  }

  // Close on outside click
  useEffect(() => {
    const handler = (e) => {
      if (!inputRef.current?.closest('[data-stocksearch]')?.contains(e.target)) {
        setOpen(false)
        setQuery('')
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <div data-stocksearch style={{ position: 'relative', width: '100%' }}>
      {/* Trigger / display */}
      {!open ? (
        <button
          onClick={() => { if (!disabled) { setOpen(true); setTimeout(() => inputRef.current?.focus(), 10) } }}
          disabled={disabled}
          style={{
            width: '100%', background: '#0f172a', border: '1px solid #1e293b',
            borderRadius: 8, padding: '10px 12px', color: '#e2e8f0',
            fontSize: 14, textAlign: 'left', cursor: disabled ? 'not-allowed' : 'pointer',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            opacity: disabled ? 0.5 : 1,
          }}
        >
          <span>{currentLabel || 'Select stock…'}</span>
          <span style={{ color: '#475569', fontSize: 11 }}>▾</span>
        </button>
      ) : (
        <input
          ref={inputRef}
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder={`Search ${symbols.length} stocks…`}
          autoFocus
          style={{
            width: '100%', background: '#0f172a', border: '1px solid #16a34a',
            borderRadius: 8, padding: '10px 12px', color: '#e2e8f0',
            fontSize: 14, outline: 'none', boxSizing: 'border-box',
          }}
        />
      )}

      {/* Dropdown */}
      {open && (
        <div
          ref={listRef}
          style={{
            position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0,
            background: '#111827', border: '1px solid #1e293b',
            borderRadius: 8, zIndex: 100, maxHeight: 280, overflowY: 'auto',
            boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
          }}
        >
          {filtered.length === 0 && (
            <div style={{ padding: '12px 14px', color: '#475569', fontSize: 13 }}>
              No results for "{query}"
            </div>
          )}
          {filtered.map(s => (
            <div
              key={s.value}
              onMouseDown={() => select(s)}
              style={{
                padding: '9px 14px', cursor: 'pointer', fontSize: 13,
                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                background: s.value === value ? 'rgba(22,163,74,0.12)' : 'transparent',
                borderLeft: s.value === value ? '2px solid #16a34a' : '2px solid transparent',
              }}
              onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.04)'}
              onMouseLeave={e => e.currentTarget.style.background = s.value === value ? 'rgba(22,163,74,0.12)' : 'transparent'}
            >
              <span style={{ color: '#e2e8f0', fontWeight: s.value === value ? 500 : 400 }}>
                {s.label}
              </span>
              <span style={{ color: '#475569', fontSize: 11, fontFamily: 'var(--font-mono)' }}>
                {s.value.replace('.NS', '')}
              </span>
            </div>
          ))}
          {filtered.length === 80 && !query && (
            <div style={{ padding: '8px 14px', color: '#475569', fontSize: 11, borderTop: '1px solid #1e293b', textAlign: 'center' }}>
              Type to search all {symbols.length} stocks
            </div>
          )}
        </div>
      )}
    </div>
  )
}
