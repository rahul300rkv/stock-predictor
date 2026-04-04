import os, warnings, uuid
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
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
    df["RSI"]          = compute_rsi(df["Close"])
    df["MACD"], df["MACD_signal"] = compute_macd(df["Close"])
    df["MACD_Hist"]    = df["MACD"] - df["MACD_signal"]
    df["EMA_10"]       = df["Close"].ewm(span=10, adjust=False).mean()
    df["EMA_50"]       = df["Close"].ewm(span=30, adjust=False).mean()
    df["Crossover"]    = np.where(df["EMA_10"] > df["EMA_50"], 1, 0).astype(float)
    df["ATR"]          = compute_atr(df["High"], df["Low"], df["Close"])
    df["ATR_Norm"]     = df["ATR"] / df["Close"]
    df["ADX"]          = compute_adx(df["High"], df["Low"], df["Close"])
    df["Momentum"]     = df["Close"].pct_change(3)
    df["VWAP"]         = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    ll = df["Low"].rolling(14).min(); hh = df["High"].rolling(14).max()
    df["Stochastic"]   = (100 * (df["Close"] - ll) / (hh - ll).replace(0, np.nan)).fillna(50)
    df["ROC"]          = df["Close"].pct_change(5) * 100
    hh2 = df["High"].rolling(14).max(); ll2 = df["Low"].rolling(14).min()
    df["Williams_R"]   = (-100 * (hh2 - df["Close"]) / (hh2 - ll2).replace(0, np.nan)).fillna(-50)
    df["OBV"]          = (np.sign(df["Close"].diff()) * df["Volume"]).cumsum()
    df["Volatility_20"] = df["Close"].rolling(20).std()
    ma20 = df["Close"].rolling(20).mean()
    df["BB_High"]      = ma20 + 2 * df["Volatility_20"]
    df["BB_Low"]       = ma20 - 2 * df["Volatility_20"]
    for lag in range(1, 6):
        df[f"Lag_Close_{lag}"]  = df["Close"].shift(lag)
        df[f"Lag_Volume_{lag}"] = df["Volume"].shift(lag)
    df["DayOfWeek"]    = df.index.dayofweek.astype(float)
    df["Month"]        = df.index.month.astype(float)
    df["IsMonthEnd"]   = df.index.is_month_end.astype(float)
    df["IsMonthStart"] = df.index.is_month_start.astype(float)
    return df.ffill().fillna(0).dropna()

def feature_selection(df):
    corr     = df.corr()["Close"].abs()
    selected = corr[corr > 0.1].index.tolist()
    must     = ["Close","EMA_10","EMA_50","BB_Low","BB_High","ATR","VWAP",
                "Volatility_20","RSI","OBV","Lag_Close_2","Lag_Volume_2"]
    return list(set(selected + must))

def make_flat_features(scaled, n_steps, close_idx):
    """Flatten sliding windows into 2D for sklearn models."""
    X, yr = [], []
    for i in range(n_steps, len(scaled) - LOOKUP_STEP + 1):
        X.append(scaled[i - n_steps:i].flatten())
        yr.append(scaled[i + LOOKUP_STEP - 1][close_idx])
    return np.array(X), np.array(yr)

def inverse_transform(preds, col_idx, scaler):
    return np.array(preds).flatten() * scaler.scale_[col_idx] + scaler.mean_[col_idx]

def load_data(ticker):
    ticker = ticker.upper().strip()
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker = ticker + ".NS"

    df = yf.download(ticker, period="5y", interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the symbol.")

    df = df[["Open","High","Low","Close","Volume"]]
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
    yc = df_label.values[N_STEPS: N_STEPS + len(yr)]

    n = max(1, int(TEST_SIZE * len(X_flat)))
    dates = df.index[N_STEPS: N_STEPS + len(yr)]

    return (X_flat[:-n], X_flat[-n:],
            yr[:-n],     yr[-n:],
            yc[:-n],     yc[-n:],
            scaler, dates, df, cidx)

def run_job(job_id, ticker, mtype, bidir):
    try:
        jobs[job_id]["status"] = "loading_data"
        X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, scaler, dates, df, cidx = load_data(ticker)

        tscv = TimeSeriesSplit(n_splits=N_FOLDS)
        reg_preds_list = []
        clf_preds_list = []

        for fold, (ti, vi) in enumerate(tscv.split(X_tr)):
            jobs[job_id]["status"] = f"training_fold_{fold+1}_of_{N_FOLDS}"

            # Regression model
            if mtype == "GRU" or mtype == "LSTM":
                reg = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                learning_rate=0.05, random_state=42)
            elif mtype == "Conv1D":
                reg = RandomForestRegressor(n_estimators=200, max_depth=8,
                                            random_state=42, n_jobs=-1)
            else:
                reg = Ridge(alpha=1.0)

            reg.fit(X_tr[ti], yr_tr[ti])
            reg_preds_list.append(reg.predict(X_te))

            # Classification model
            if mtype in ["GRU", "LSTM"]:
                clf = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                 learning_rate=0.05, random_state=42)
            elif mtype == "Conv1D":
                clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                             random_state=42, n_jobs=-1)
            else:
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(C=1.0, max_iter=500)

            clf.fit(X_tr[ti], yc_tr[ti])
            clf_preds_list.append(clf.predict_proba(X_te)[:, 1])

        jobs[job_id]["status"] = "predicting"

        # Ensemble average across folds
        yrp = np.mean(reg_preds_list, axis=0)
        ycp = np.mean(clf_preds_list, axis=0)
        direction = (ycp > 0.5).astype(int)
        adj = yrp * (1 + 0.1 * (2 * direction - 1))

        ya = inverse_transform(adj, cidx, scaler)
        yt = inverse_transform(yr_te, cidx, scaler)

        nz   = yt != 0
        mape = float(np.mean(np.abs((yt[nz] - ya[nz]) / yt[nz])) * 100) if nz.any() else 0
        rmse = float(np.sqrt(np.mean((yt - ya) ** 2)))
        da   = float(np.mean(np.sign(yt[1:] - yt[:-1]) == np.sign(ya[1:] - ya[:-1])) * 100)

        # Future prediction
        fut_reg = np.mean([reg.predict(X_te[-1:]) for reg in
                           [GradientBoostingRegressor(n_estimators=200, max_depth=4,
                            learning_rate=0.05, random_state=42).fit(X_tr, yr_tr)]], axis=0)
        fut_clf_prob = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                       learning_rate=0.05, random_state=42).fit(X_tr, yc_tr).predict_proba(X_te[-1:])[:, 1]
        fut_dir = (fut_clf_prob > 0.5).astype(int)
        fut_adj = fut_reg * (1 + 0.1 * (2 * fut_dir - 1))
        fp  = float(inverse_transform(fut_adj, cidx, scaler)[0])
        fd  = (dates[-1] + BDay(LOOKUP_STEP)).strftime("%d-%b-%Y")

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
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

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
