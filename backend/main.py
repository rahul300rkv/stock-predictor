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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional
from pandas.tseries.offsets import BDay

warnings.filterwarnings("ignore")

N_STEPS, LOOKUP_STEP, TEST_SIZE = 10, 1, 0.1
BATCH_SIZE, EPOCHS, PATIENCE    = 32, 30, 5
LEARNING_RATE, N_FOLDS          = 0.0001, 3
THRESHOLD                       = 0.005

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
    tr = pd.concat([high-low,
                    (high-close.shift()).abs(),
                    (low-close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()

def compute_adx(high, low, close, period=14):
    atr      = compute_atr(high, low, close, period)
    up, dn   = high - high.shift(), low.shift() - low
    dmp      = up.where((up > dn) & (up > 0), 0.0)
    dmm      = dn.where((dn > up) & (dn > 0), 0.0)
    dip      = 100 * dmp.ewm(alpha=1/period, min_periods=period).mean() / atr.replace(0, np.nan)
    dim      = 100 * dmm.ewm(alpha=1/period, min_periods=period).mean() / atr.replace(0, np.nan)
    dx       = ((dip - dim).abs() / (dip + dim).replace(0, np.nan)) * 100
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

def inverse_transform(preds, col_idx, scaler):
    return np.array(preds).flatten() * scaler.scale_[col_idx] + scaler.mean_[col_idx]

def load_data(ticker):
    """Download 5y daily OHLCV via yfinance (works on Render IPs)."""
    ticker = ticker.upper().strip()
    # Ensure .NS suffix for NSE stocks
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker = ticker + ".NS"

    df = yf.download(ticker, period="5y", interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the symbol is valid.")

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
    scaler   = StandardScaler()
    scaled   = scaler.fit_transform(df_feat.values)
    cidx     = fcols.index("Close")

    X, yr, yc = [], [], []
    lab = df_label.values
    for i in range(N_STEPS, len(scaled) - LOOKUP_STEP + 1):
        X.append(scaled[i-N_STEPS:i])
        yr.append(scaled[i+LOOKUP_STEP-1][cidx])
        yc.append(lab[i+LOOKUP_STEP-1])

    X, yr, yc = np.array(X), np.array(yr), np.array(yc)
    n = max(1, int(TEST_SIZE * len(X)))
    return X[:-n], X[-n:], yr[:-n], yr[-n:], yc[:-n], yc[-n:], \
           scaler, df.index[-len(yr):], df, cidx

def custom_loss(y_true, y_pred):
    mse   = tf.reduce_mean(tf.square(y_true - y_pred))
    dt    = tf.sign(y_true[1:] - y_true[:-1])
    dp    = tf.sign(y_pred[1:] - y_pred[:-1])
    dloss = tf.reduce_mean(tf.cast(tf.not_equal(dt, dp), tf.float32))
    return mse + 1.5 * dloss

def build_model(mtype, shape, bidir=False, classify=False):
    inp = keras.Input(shape=shape)
    if mtype == "GRU":    base = layers.GRU(64, return_sequences=True)
    elif mtype == "LSTM": base = layers.LSTM(64, return_sequences=True)
    elif mtype == "Conv1D": base = layers.Conv1D(64, 3, activation="relu", padding="same")
    else:                 base = layers.TimeDistributed(layers.Dense(64, activation="relu"))
    x = Bidirectional(base)(inp) if (bidir and mtype in ["GRU","LSTM"]) else base(inp)
    if mtype != "Conv1D":
        x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MultiHeadAttention(num_heads=4, key_dim=8)(x, x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
    out = layers.Dense(1, activation="sigmoid" if classify else None)(x)
    m   = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
              loss="binary_crossentropy" if classify else custom_loss,
              metrics=["accuracy"] if classify else [])
    return m

def run_job(job_id, ticker, mtype, bidir):
    try:
        jobs[job_id]["status"] = "loading_data"
        X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te, scaler, dates, df, cidx = load_data(ticker)

        cb = [EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
              ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6)]
        tscv     = TimeSeriesSplit(n_splits=N_FOLDS)
        reg_ms, clf_ms = [], []

        for fold, (ti, vi) in enumerate(tscv.split(X_tr)):
            jobs[job_id]["status"] = f"training_fold_{fold+1}_of_{N_FOLDS}"
            rm = build_model(mtype, X_tr.shape[1:], bidir, classify=False)
            rm.fit(X_tr[ti], yr_tr[ti], validation_data=(X_tr[vi], yr_tr[vi]),
                   epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=0)
            reg_ms.append(rm)
            cm = build_model(mtype, X_tr.shape[1:], bidir, classify=True)
            cm.fit(X_tr[ti], yc_tr[ti], validation_data=(X_tr[vi], yc_tr[vi]),
                   epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=0)
            clf_ms.append(cm)

        jobs[job_id]["status"] = "predicting"
        w   = np.ones(len(reg_ms)) / len(reg_ms)
        yrp = np.sum([w[i]*reg_ms[i].predict(X_te, verbose=0)
                      for i in range(len(reg_ms))], axis=0)
        ycp = np.mean([m.predict(X_te, verbose=0) for m in clf_ms], axis=0)
        adj = yrp * (1 + 0.1 * (2*(ycp > 0.5).astype(int) - 1))

        ya  = inverse_transform(adj, cidx, scaler)
        yt  = inverse_transform(yr_te.reshape(-1,1), cidx, scaler)
        nz  = yt != 0
        mape= float(np.mean(np.abs((yt[nz]-ya[nz])/yt[nz]))*100) if nz.any() else 0
        rmse= float(np.sqrt(np.mean((yt-ya)**2)))
        da  = float(np.mean(np.sign(yt[1:]-yt[:-1]) == np.sign(ya[1:]-ya[:-1]))*100)

        frp = np.sum([w[i]*reg_ms[i].predict(X_te[-1:], verbose=0)
                      for i in range(len(reg_ms))], axis=0)
        fcp = np.mean([m.predict(X_te[-1:], verbose=0) for m in clf_ms], axis=0)
        fad = frp * (1 + 0.1 * (2*(fcp > 0.5).astype(int) - 1))
        fp  = float(inverse_transform(fad, cidx, scaler)[0])
        fd  = (dates[-1] + BDay(LOOKUP_STEP)).strftime("%d-%b-%Y")

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "ticker":       ticker,
            "model_type":   mtype,
            "bidirectional": bidir,
            "future_price": round(fp, 2),
            "future_date":  fd,
            "current_price": round(float(yt[-1]), 2),
            "mape":  round(mape, 2),
            "rmse":  round(rmse, 2),
            "directional_accuracy": round(da, 2),
            "chart_dates":  [str(d.date()) for d in dates[-len(yt):]],
            "chart_actual": [round(float(v), 2) for v in yt],
            "chart_pred":   [round(float(v), 2) for v in ya],
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
    return {"message": "Stock Predictor API is running. Visit /docs for API reference."}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

class PredictReq(BaseModel):
    ticker:       str
    model_type:   str  = "GRU"
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
