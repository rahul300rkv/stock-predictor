import os, warnings, uuid, requests
import numpy as np
import pandas as pd
from datetime import datetime, date
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from pandas.tseries.offsets import BDay

warnings.filterwarnings("ignore")

N_STEPS     = 10
LOOKUP_STEP = 1
TEST_SIZE   = 0.1
N_FOLDS     = 3
THRESHOLD   = 0.005

app = FastAPI(title="Stock Predictor API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

jobs: dict = {}

# ── Twelve Data key rotation ──────────────────────────────────────────────────
# Add as many keys as you have accounts - they rotate automatically
TD_KEYS = [k.strip() for k in [
    os.environ.get("TWELVEDATA_API_KEY_1", ""),
    os.environ.get("TWELVEDATA_API_KEY_2", ""),
    os.environ.get("TWELVEDATA_API_KEY_3", ""),  # add more if needed
] if k.strip()]

_key_index = 0  # current active key

def get_next_key():
    """Rotate to next available key."""
    global _key_index
    if not TD_KEYS:
        raise ValueError("No Twelve Data API keys set. Add TWELVEDATA_API_KEY_1 and TWELVEDATA_API_KEY_2 in Render environment variables.")
    _key_index = (_key_index + 1) % len(TD_KEYS)
    return TD_KEYS[_key_index]

def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    """
    Fetch 5 years of daily OHLCV from Twelve Data.
    Automatically rotates API keys if one hits rate limit.
    symbol: NSE symbol e.g. 'RELIANCE.NS' or 'RELIANCE'
    """
    clean = symbol.upper().replace(".NS", "").replace(".BO", "").replace(".BSE", "")

    last_error = None
    # Try each key up to 2 full rotations
    attempts = len(TD_KEYS) * 2 if TD_KEYS else 1

    for attempt in range(attempts):
        key = TD_KEYS[_key_index % len(TD_KEYS)] if TD_KEYS else ""
        try:
            resp = requests.get(
                "https://api.twelvedata.com/time_series",
                params={
                    "symbol":     clean,
                    "exchange":   "NSE",
                    "interval":   "1day",
                    "outputsize": 1260,      # ~5 years trading days
                    "apikey":     key,
                    "format":     "JSON",
                    "order":      "ASC",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            # Rate limit hit → rotate key and retry
            if data.get("code") in [429, 400] or "rate limit" in str(data.get("message","")).lower():
                print(f"Key {_key_index} rate limited, rotating...")
                get_next_key()
                last_error = data.get("message", "Rate limit")
                continue

            if data.get("status") == "error":
                raise ValueError(f"Twelve Data error: {data.get('message', 'Unknown')}")

            values = data.get("values")
            if not values:
                raise ValueError(f"No data returned for {clean} on NSE.")

            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            df = df.rename(columns={
                "open": "Open", "high": "High",
                "low":  "Low",  "close": "Close", "volume": "Volume"
            })
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

            if df.empty:
                raise ValueError(f"Empty data for {clean}")

            print(f"Fetched {len(df)} rows for {clean} using key index {_key_index}")
            return df

        except ValueError:
            raise
        except Exception as e:
            last_error = str(e)
            get_next_key()
            continue

    raise ValueError(f"All Twelve Data keys failed. Last error: {last_error}")

# ── Indicators ────────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    ag    = gain.ewm(alpha=1/period, min_periods=period).mean()
    al    = loss.ewm(alpha=1/period, min_periods=period).mean()
    return (100 - 100 / (1 + ag / al.replace(0, np.nan))).fillna(50)

def compute_macd(series, fast=12, slow=26, signal=9):
    macd = series.ewm(span=fast, adjust=False).mean() - series.ewm(span=slow, adjust=False).mean()
    return macd, macd.ewm(span=signal, adjust=False).mean()

def compute_atr(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()

def compute_adx(high, low, close, period=14):
    atr = compute_atr(high, low, close, period)
    up, dn = high - high.shift(), low.shift() - low
    dmp = up.where((up > dn) & (up > 0), 0.0)
    dmm = dn.where((dn > up) & (dn > 0), 0.0)
    dip = 100 * dmp.ewm(alpha=1/period, min_periods=period).mean() / atr.replace(0, np.nan)
    dim = 100 * dmm.ewm(alpha=1/period, min_periods=period).mean() / atr.replace(0, np.nan)
    dx  = ((dip - dim).abs() / (dip + dim).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1/period, min_periods=period).mean().fillna(0)

def add_indicators(df):
    df = df.copy()
    df["RSI"]           = compute_rsi(df["Close"])
    df["MACD"], df["MACD_signal"] = compute_macd(df["Close"])
    df["MACD_Hist"]     = df["MACD"] - df["MACD_signal"]
    df["EMA_10"]        = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_50"]        = df["Close"].ewm(span=30, adjust=False).mean()
    df["Crossover"]     = np.where(df["EMA_10"] > df["EMA_50"], 1, 0).astype(float)
    df["ATR"]           = compute_atr(df["High"], df["Low"], df["Close"])
    df["ATR_Norm"]      = df["ATR"] / df["Close"]
    df["ADX"]           = compute_adx(df["High"], df["Low"], df["Close"])
    df["Momentum"]      = df["Close"].pct_change(3)
    df["VWAP"]          = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    ll = df["Low"].rolling(14).min(); hh = df["High"].rolling(14).max()
    df["Stochastic"]    = (100 * (df["Close"] - ll) / (hh - ll).replace(0, np.nan)).fillna(50)
    df["ROC"]           = df["Close"].pct_change(5) * 100
    hh2 = df["High"].rolling(14).max(); ll2 = df["Low"].rolling(14).min()
    df["Williams_R"]    = (-100 * (hh2 - df["Close"]) / (hh2 - ll2).replace(0, np.nan)).fillna(-50)
    df["OBV"]           = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
    df["Volatility_20"] = df["Close"].rolling(20).std()
    ma20 = df["Close"].rolling(20).mean()
    df["BB_High"]       = ma20 + 2 * df["Volatility_20"]
    df["BB_Low"]        = ma20 - 2 * df["Volatility_20"]
    for lag in range(1, 6):
        df[f"Lag_Close_{lag}"]  = df["Close"].shift(lag)
        df[f"Lag_Volume_{lag}"] = df["Volume"].shift(lag)
    df["DayOfWeek"]     = df.index.dayofweek.astype(float)
    df["Month"]         = df.index.month.astype(float)
    df["IsMonthEnd"]    = df.index.is_month_end.astype(float)
    df["IsMonthStart"]  = df.index.is_month_start.astype(float)
    return df.ffill().fillna(0).dropna()

def feature_selection(df):
    corr     = df.corr()["Close"].abs()
    selected = corr[corr > 0.1].index.tolist()
    must     = ["Close","EMA_10","EMA_50","BB_Low","BB_High","ATR","VWAP",
                "Volatility_20","RSI","OBV","Lag_Close_2","Lag_Volume_2"]
    return list(set(selected + must))

def make_flat_features(scaled, n_steps, close_idx):
    X, yr = [], []
    for i in range(n_steps, len(scaled) - LOOKUP_STEP + 1):
        X.append(scaled[i - n_steps:i].flatten())
        yr.append(scaled[i + LOOKUP_STEP - 1][close_idx])
    return np.array(X), np.array(yr)

def inverse_transform(preds, col_idx, scaler):
    return np.array(preds).flatten() * scaler.scale_[col_idx] + scaler.mean_[col_idx]

def load_data(ticker):
    ticker = ticker.upper().strip()
    df = fetch_ohlcv(ticker)
    df = df[df["Close"].between(df["Close"].quantile(0.005),
                                df["Close"].quantile(0.995))]
    df = add_indicators(df)

    future = df["Close"].shift(-LOOKUP_STEP)
    df["Price_Direction"] = ((future - df["Close"]) / df["Close"] > THRESHOLD).astype(int)

    fcols = feature_selection(df)
    if "Price_Direction" in fcols:
        fcols.remove("Price_Direction")

    df_feat  = df[fcols].dropna()
    df_label = df.loc[df_feat.index, "Price_Direction"]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_feat.values)
    cidx   = fcols.index("Close")

    X_flat, yr = make_flat_features(scaled, N_STEPS, cidx)
    yc    = df_label.values[N_STEPS: N_STEPS + len(yr)]
    dates = df_feat.index[N_STEPS: N_STEPS + len(yr)]

    n = max(1, int(TEST_SIZE * len(X_flat)))
    return (X_flat[:-n], X_flat[-n:],
            yr[:-n],     yr[-n:],
            yc[:-n],     yc[-n:],
            scaler, dates, df, cidx)

def get_models(mtype):
    if mtype in ["GRU", "LSTM"]:
        reg = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                        learning_rate=0.05, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                         learning_rate=0.05, random_state=42)
    elif mtype == "Conv1D":
        reg = RandomForestRegressor(n_estimators=200, max_depth=8,
                                    random_state=42, n_jobs=-1)
        clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                     random_state=42, n_jobs=-1)
    else:
        reg = Ridge(alpha=1.0)
        clf = LogisticRegression(C=1.0, max_iter=500)
    return reg, clf

def run_job(job_id, ticker, mtype, bidir):
    try:
        jobs[job_id]["status"] = "loading_data"
        X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, scaler, dates, df, cidx = load_data(ticker)

        tscv = TimeSeriesSplit(n_splits=N_FOLDS)
        reg_preds, clf_preds = [], []
        final_reg = final_clf = None

        for fold, (ti, vi) in enumerate(tscv.split(X_tr)):
            jobs[job_id]["status"] = f"training_fold_{fold+1}_of_{N_FOLDS}"
            reg, clf = get_models(mtype)
            reg.fit(X_tr[ti], yr_tr[ti])
            clf.fit(X_tr[ti], yc_tr[ti])
            reg_preds.append(reg.predict(X_te))
            clf_preds.append(clf.predict_proba(X_te)[:, 1])
            final_reg, final_clf = reg, clf  # keep last fold for future pred

        jobs[job_id]["status"] = "predicting"
        yrp = np.mean(reg_preds, axis=0)
        ycp = np.mean(clf_preds, axis=0)
        adj = yrp * (1 + 0.1 * (2 * (ycp > 0.5).astype(int) - 1))

        ya = inverse_transform(adj, cidx, scaler)
        yt = inverse_transform(yr_te, cidx, scaler)

        nz   = yt != 0
        mape = float(np.mean(np.abs((yt[nz] - ya[nz]) / yt[nz])) * 100) if nz.any() else 0
        rmse = float(np.sqrt(np.mean((yt - ya) ** 2)))
        da   = float(np.mean(np.sign(yt[1:] - yt[:-1]) == np.sign(ya[1:] - ya[:-1])) * 100)

        # Future price
        fut_reg = final_reg.predict(X_te[-1:])
        fut_clf = final_clf.predict_proba(X_te[-1:])[:, 1]
        fut_adj = fut_reg * (1 + 0.1 * (2 * (fut_clf > 0.5).astype(int) - 1))
        fp = float(inverse_transform(fut_adj, cidx, scaler)[0])
        fd = (dates[-1] + BDay(LOOKUP_STEP)).strftime("%d-%b-%Y")

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "ticker":        ticker,
            "model_type":    mtype,
            "bidirectional": bidir,
            "future_price":  round(fp, 2),
            "future_date":   fd,
            "current_price": round(float(yt[-1]), 2),
            "mape":          round(mape, 2),
            "rmse":          round(rmse, 2),
            "directional_accuracy": round(da, 2),
            "chart_dates":   [str(d.date()) for d in dates[-len(yt):]],
            "chart_actual":  [round(float(v), 2) for v in yt],
            "chart_pred":    [round(float(v), 2) for v in ya],
            "ohlcv": [
                {"date":   str(idx.date()),
                 "open":   round(float(r["Open"]),  2),
                 "high":   round(float(r["High"]),  2),
                 "low":    round(float(r["Low"]),   2),
                 "close":  round(float(r["Close"]), 2),
                 "volume": int(r["Volume"])}
                for idx, r in df.tail(30).iterrows()
            ],
        }
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Stock Predictor API running. Visit /docs for reference."}

@app.get("/health")
def health():
    active_keys = len(TD_KEYS)
    return {"status": "ok", "api_keys_loaded": active_keys,
            "timestamp": datetime.utcnow().isoformat()}

class PredictReq(BaseModel):
    ticker:        str
    model_type:    str  = "GRU"
    bidirectional: bool = False

@app.post("/predict")
def start_predict(req: PredictReq, bg: BackgroundTasks):
    jid = str(uuid.uuid4())
    jobs[jid] = {"status": "queued", "result": None, "error": None}
    bg.add_task(run_job, jid,
                req.ticker.upper().strip(),
                req.model_type,
                req.bidirectional)
    return {"job_id": jid}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return {"job_id": job_id, "status": j["status"],
            "result": j["result"], "error": j.get("error")}

def load_symbols_from_csv(path: str = "nifty.csv") -> list:
    try:
        df = pd.read_csv(path)
        syms = df.iloc[:, 0].dropna().str.strip().tolist()
        return [{"label": s.replace(".NS","").replace(".BO",""), "value": s}
                for s in syms if s and s != "Symbol"]
    except Exception:
        return [{"label": "RELIANCE", "value": "RELIANCE.NS"}]

_SYMBOLS = load_symbols_from_csv()

@app.get("/symbols")
def symbols(q: str = ""):
    data = _SYMBOLS
    if q:
        ql = q.lower()
        data = [s for s in data
                if ql in s["label"].lower() or ql in s["value"].lower()]
    return {"symbols": data, "total": len(data)}
