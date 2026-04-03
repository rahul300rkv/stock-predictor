'use client'
import { useEffect, useRef } from 'react'
import {
  Chart, LineElement, PointElement, LinearScale,
  CategoryScale, Tooltip, Legend, Filler,
} from 'chart.js'
import { Line } from 'react-chartjs-2'

Chart.register(LineElement, PointElement, LinearScale, CategoryScale, Tooltip, Legend, Filler)

export default function PredictionChart({ dates, actual, predicted }) {
  const step = Math.max(1, Math.floor(dates.length / 10))
  const labels = dates.map((d, i) => i % step === 0 ? d : '')

  const data = {
    labels,
    datasets: [
      {
        label: 'Actual',
        data: actual,
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56,189,248,0.06)',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      },
      {
        label: 'Predicted',
        data: predicted,
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245,158,11,0.06)',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
        borderDash: [5, 3],
      },
    ],
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        labels: {
          color: '#94a3b8', font: { size: 12 },
          boxWidth: 24, boxHeight: 2,
        },
      },
      tooltip: {
        backgroundColor: '#1e293b',
        titleColor: '#94a3b8',
        bodyColor: '#e2e8f0',
        borderColor: '#334155',
        borderWidth: 1,
        callbacks: {
          label: ctx => ` ₹${ctx.parsed.y.toLocaleString('en-IN')}`,
        },
      },
    },
    scales: {
      x: {
        ticks: { color: '#475569', font: { size: 11 }, maxRotation: 0 },
        grid:  { color: 'rgba(30,41,59,0.8)' },
      },
      y: {
        ticks: {
          color: '#475569', font: { size: 11 },
          callback: v => `₹${v.toLocaleString('en-IN')}`,
        },
        grid: { color: 'rgba(30,41,59,0.8)' },
      },
    },
  }

  return (
    <div style={{ height: 320 }}>
      <Line data={data} options={options} />
    </div>
  )
}
