import './globals.css'
import { DM_Sans, DM_Mono } from 'next/font/google'

const dmSans = DM_Sans({ subsets: ['latin'], variable: '--font-sans' })
const dmMono = DM_Mono({ subsets: ['latin'], weight: ['400','500'], variable: '--font-mono' })

export const metadata = {
  title: 'StockSeer — AI Stock Predictor',
  description: 'Deep learning predictions for Indian stocks using GRU/LSTM models',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${dmSans.variable} ${dmMono.variable}`}>
      <body>{children}</body>
    </html>
  )
}
