# StockSeer — AI Stock Predictor

Deep learning predictions for Nifty 50 Indian stocks using GRU/LSTM/Conv1D models with attention.

## Project Structure

```
stock-predictor/
├── backend/        → Hugging Face Spaces (FastAPI + Docker)
└── frontend/       → Vercel (Next.js 14)
```

---

## 1. Deploy Backend → Hugging Face Spaces

### a) Create a new Space
1. Go to https://huggingface.co/new-space
2. Name it `stock-predictor-api`
3. Select **Docker** as SDK
4. Set visibility to **Public**

### b) Push backend code
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/stock-predictor-api
cp -r backend/* stock-predictor-api/
cd stock-predictor-api
git add .
git commit -m "Initial deploy"
git push
```

Your API will be live at:
`https://YOUR_USERNAME-stock-predictor-api.hf.space`

---

## 2. Deploy Frontend → Vercel

### a) Set environment variable
Copy `frontend/.env.example` to `frontend/.env.local`:
```bash
NEXT_PUBLIC_API_URL=https://YOUR_USERNAME-stock-predictor-api.hf.space
```

### b) Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/stock-predictor.git
git push -u origin main
```

### c) Deploy to Vercel
1. Go to https://vercel.com/new
2. Import your GitHub repo
3. Set **Root Directory** to `frontend`
4. Add environment variable:
   - `NEXT_PUBLIC_API_URL` = your HuggingFace Space URL
5. Click Deploy

---

## Usage Notes

- First prediction takes **5–15 minutes** (trains 3 folds × 2 models = 6 neural networks)
- HF free tier is CPU only — GPU upgrade speeds this up 10×
- Models train fresh on each request using 5 years of live Yahoo Finance data

## Models Available

| Model   | Description |
|---------|-------------|
| GRU     | Gated Recurrent Unit (fastest) |
| LSTM    | Long Short-Term Memory |
| Conv1D  | Temporal Convolution |
| Dense   | Fully connected baseline |

All models include: Multi-Head Attention, Batch Normalization, Dropout, L2 regularization.
