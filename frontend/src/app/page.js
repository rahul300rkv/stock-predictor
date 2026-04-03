'use client'
import { useState, useEffect, useRef, useCallback } from 'react'
import PredictionChart from '../components/PredictionChart'
import StockSearch from '../components/StockSearch'
import MetricCard from '../components/MetricCard'
import OHLCVTable from '../components/OHLCVTable'

const API = process.env.NEXT_PUBLIC_API_URL || 'https://your-space.hf.space'

const STATUS_LABELS = {
  queued:       'Queued…',
  loading_data: 'Downloading market data…',
  training_fold_1_of_3: 'Training fold 1 of 3…',
  training_fold_2_of_3: 'Training fold 2 of 3…',
  training_fold_3_of_3: 'Training fold 3 of 3…',
  predicting:   'Running predictions…',
  done:         'Complete',
  error:        'Error',
}

export default function Home() {
  const [symbols, setSymbols]     = useState([])
  const [ticker,  setTicker]      = useState('')
  const [model,   setModel]       = useState('GRU')
  const [bidir,   setBidir]       = useState(false)
  const [jobId,   setJobId]       = useState(null)
  const [status,  setStatus]      = useState(null)
  const [result,  setResult]      = useState(null)
  const [error,   setError]       = useState(null)
  const [loading, setLoading]     = useState(false)
  const pollRef = useRef(null)

  useEffect(() => {
    fetch(`${API}/symbols`)
      .then(r => r.json())
      .then(d => { setSymbols(d.symbols); setTicker(d.symbols[0]?.value || '') })
      .catch(() => {})
  }, [])

  const poll = useCallback((jid) => {
    pollRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API}/status/${jid}`)
        const d = await r.json()
        setStatus(d.status)
        if (d.status === 'done') {
          clearInterval(pollRef.current)
          setResult(d.result)
          setLoading(false)
        } else if (d.status === 'error') {
          clearInterval(pollRef.current)
          setError(d.error || 'Unknown error')
          setLoading(false)
        }
      } catch (e) {
        clearInterval(pollRef.current)
        setError('Lost connection to API')
        setLoading(false)
      }
    }, 3000)
  }, [])

  const handlePredict = async () => {
    if (!ticker) return
    clearInterval(pollRef.current)
    setLoading(true)
    setResult(null)
    setError(null)
    setStatus('queued')
    try {
      const r = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker, model_type: model, bidirectional: bidir }),
      })
      const d = await r.json()
      setJobId(d.job_id)
      poll(d.job_id)
    } catch (e) {
      setError('Could not reach the API. Is the HuggingFace Space awake?')
      setLoading(false)
    }
  }

  useEffect(() => () => clearInterval(pollRef.current), [])

  const priceChange = result
    ? ((result.future_price - result.current_price) / result.current_price * 100).toFixed(2)
    : null
  const bullish = priceChange > 0

  return (
    <div className="min-h-screen" style={{ background: '#0a0f1a' }}>
      {/* ── Header ─────────────────────────────────────────────── */}
      <header style={{
        borderBottom: '1px solid #1e293b',
        background: 'rgba(10,15,26,0.95)',
        backdropFilter: 'blur(12px)',
        position: 'sticky', top: 0, zIndex: 50,
      }}>
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div style={{
              width: 36, height: 36, borderRadius: 10,
              background: 'linear-gradient(135deg,#16a34a,#0d9488)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 18,
            }}>📈</div>
            <div>
              <div style={{ fontWeight: 600, fontSize: 16, letterSpacing: '-0.02em', color: '#f1f5f9' }}>
                StockSeer
              </div>
              <div style={{ fontSize: 11, color: '#64748b', letterSpacing: '0.05em' }}>
                AI · NIFTY STOCKS
              </div>
            </div>
          </div>
          <div style={{ fontSize: 12, color: '#475569' }}>
            Powered by TensorFlow · {model}{bidir ? ' Bidir' : ''}
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-10">
        {/* ── Control Panel ───────────────────────────────────── */}
        <div style={{
          background: '#111827',
          border: '1px solid #1e293b',
          borderRadius: 16,
          padding: '24px 28px',
          marginBottom: 32,
        }}>
          <div style={{ fontSize: 12, fontWeight: 500, color: '#64748b', letterSpacing: '0.08em', marginBottom: 20 }}>
            CONFIGURE PREDICTION
          </div>
          <div className="flex flex-wrap gap-4 items-end">
            {/* Stock */}
            <div className="flex-1" style={{ minWidth: 240 }}>
              <label style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 8 }}>
                Stock <span style={{ color: '#475569' }}>({symbols.length} available)</span>
              </label>
              <StockSearch
                symbols={symbols}
                value={ticker}
                onChange={setTicker}
                disabled={loading}
              />
            </div>

            {/* Model */}
            <div style={{ minWidth: 140 }}>
              <label style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 8 }}>Model</label>
              <select
                value={model}
                onChange={e => setModel(e.target.value)}
                disabled={loading}
                style={{
                  width: '100%', background: '#0f172a', border: '1px solid #1e293b',
                  borderRadius: 8, padding: '10px 12px', color: '#e2e8f0', fontSize: 14,
                }}
              >
                {['GRU','LSTM','Conv1D','Dense'].map(m => <option key={m}>{m}</option>)}
              </select>
            </div>

            {/* Bidir toggle */}
            <div style={{ minWidth: 130 }}>
              <label style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 8 }}>Bidirectional</label>
              <button
                onClick={() => setBidir(b => !b)}
                disabled={loading || !['GRU','LSTM'].includes(model)}
                style={{
                  width: '100%', padding: '10px 12px', borderRadius: 8, fontSize: 14, cursor: 'pointer',
                  border: '1px solid',
                  borderColor: bidir ? '#16a34a' : '#1e293b',
                  background:  bidir ? 'rgba(22,163,74,0.12)' : '#0f172a',
                  color:       bidir ? '#4ade80' : '#64748b',
                  fontWeight:  500,
                  opacity: !['GRU','LSTM'].includes(model) ? 0.4 : 1,
                }}
              >
                {bidir ? '✓ On' : 'Off'}
              </button>
            </div>

            {/* Run */}
            <button
              onClick={handlePredict}
              disabled={loading || !ticker}
              style={{
                padding: '10px 28px', borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: 'pointer',
                background: loading ? '#1e293b' : 'linear-gradient(135deg,#16a34a,#0d9488)',
                color: loading ? '#64748b' : '#fff',
                border: 'none',
                minWidth: 140,
                transition: 'all 0.2s',
              }}
            >
              {loading ? 'Running…' : '▶ Predict'}
            </button>
          </div>

          {/* Status bar */}
          {(loading || error) && (
            <div style={{ marginTop: 20, display: 'flex', alignItems: 'center', gap: 10 }}>
              {loading && (
                <>
                  <div className="pulse-dot" style={{
                    width: 8, height: 8, borderRadius: '50%', background: '#16a34a',
                  }} />
                  <span style={{ fontSize: 13, color: '#94a3b8' }}>
                    {STATUS_LABELS[status] || status || 'Processing…'}
                  </span>
                </>
              )}
              {error && (
                <span style={{ fontSize: 13, color: '#f87171' }}>⚠ {error}</span>
              )}
            </div>
          )}
        </div>

        {/* ── Results ─────────────────────────────────────────── */}
        {result && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
            {/* Prediction hero */}
            <div style={{
              background: '#111827', border: '1px solid #1e293b',
              borderRadius: 16, padding: '28px 32px',
              display: 'flex', flexWrap: 'wrap', alignItems: 'center',
              justifyContent: 'space-between', gap: 24,
            }}>
              <div>
                <div style={{ fontSize: 12, color: '#64748b', letterSpacing: '0.08em', marginBottom: 8 }}>
                  PREDICTED PRICE · {result.future_date}
                </div>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 16 }}>
                  <span style={{ fontSize: 48, fontWeight: 700, color: '#f1f5f9', letterSpacing: '-0.04em' }}>
                    ₹{result.future_price.toLocaleString('en-IN')}
                  </span>
                  <span style={{
                    fontSize: 18, fontWeight: 600,
                    color: bullish ? '#4ade80' : '#f87171',
                  }}>
                    {bullish ? '▲' : '▼'} {Math.abs(priceChange)}%
                  </span>
                </div>
                <div style={{ fontSize: 13, color: '#64748b', marginTop: 6 }}>
                  Current: ₹{result.current_price.toLocaleString('en-IN')} · {result.ticker}
                </div>
              </div>
              <div style={{
                display: 'inline-flex', alignItems: 'center', gap: 8,
                padding: '12px 20px', borderRadius: 10,
                background: bullish ? 'rgba(22,163,74,0.1)' : 'rgba(220,38,38,0.1)',
                border: `1px solid ${bullish ? 'rgba(22,163,74,0.3)' : 'rgba(220,38,38,0.3)'}`,
              }}>
                <span style={{ fontSize: 28 }}>{bullish ? '🟢' : '🔴'}</span>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 600, color: bullish ? '#4ade80' : '#f87171' }}>
                    {bullish ? 'Bullish' : 'Bearish'}
                  </div>
                  <div style={{ fontSize: 11, color: '#64748b' }}>Signal</div>
                </div>
              </div>
            </div>

            {/* Metrics */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 16 }}>
              <MetricCard label="MAPE"      value={`${result.mape}%`}   sub="Mean Abs % Error" />
              <MetricCard label="RMSE"      value={`₹${result.rmse}`}   sub="Root Mean Sq Error" />
              <MetricCard label="Dir. Acc." value={`${result.directional_accuracy}%`} sub="Directional Accuracy" accent={result.directional_accuracy > 55} />
              <MetricCard label="Model"     value={result.model_type}   sub={result.bidirectional ? 'Bidirectional' : 'Unidirectional'} />
            </div>

            {/* Chart */}
            <div style={{
              background: '#111827', border: '1px solid #1e293b',
              borderRadius: 16, padding: '24px 28px',
            }}>
              <div style={{ fontSize: 12, fontWeight: 500, color: '#64748b', letterSpacing: '0.08em', marginBottom: 20 }}>
                ACTUAL vs PREDICTED (TEST SET)
              </div>
              <PredictionChart
                dates={result.chart_dates}
                actual={result.chart_actual}
                predicted={result.chart_pred}
              />
            </div>

            {/* OHLCV */}
            <div style={{
              background: '#111827', border: '1px solid #1e293b',
              borderRadius: 16, padding: '24px 28px',
            }}>
              <div style={{ fontSize: 12, fontWeight: 500, color: '#64748b', letterSpacing: '0.08em', marginBottom: 20 }}>
                RECENT OHLCV (LAST 30 DAYS)
              </div>
              <OHLCVTable data={result.ohlcv} />
            </div>
          </div>
        )}

        {/* Empty state */}
        {!result && !loading && (
          <div style={{ textAlign: 'center', padding: '80px 0', color: '#334155' }}>
            <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.4 }}>📊</div>
            <div style={{ fontSize: 16, fontWeight: 500 }}>Select a stock and click Predict</div>
            <div style={{ fontSize: 13, marginTop: 8 }}>Training takes 5–15 minutes on CPU</div>
          </div>
        )}
      </main>
    </div>
  )
}
