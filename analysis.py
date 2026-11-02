# -*- coding: utf-8 -*-
"""Analysis logic: seasonal climatology, drought status and 1-day-ahead forecast."""

import json
from functools import lru_cache
from pathlib import Path
from xgboost import XGBRegressor

import numpy as np
import pandas as pd

CLIM_FILE = Path("data/climatology.csv")
MODEL_DIR = Path("models")
LEVELS = [2, 5, 10, 20, 30, 50, 70, 90, 95]   # percentile levels stored per station / day-of-year
CLIM_WINDOW = 7                                  # +/- days around the day-of-year
BASE_FEATURES = ["doy_sin", "doy_cos", "flow_lag1", "flow_lag7"]


# --------------------------------------------------------------------------- features
def doy365(index):
    return np.minimum(pd.DatetimeIndex(index).dayofyear, 365)


def seasonal_features(index):
    doy = doy365(index)
    return np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)


def build_training_frame(history: pd.DataFrame) -> pd.DataFrame:
    """One row per station-day with lag features computed on a continuous daily grid,
    so that lag1 / lag7 always mean exactly 1 / 7 calendar days earlier."""
    frames = []
    for sid, g in history.groupby("station_id"):
        s = pd.Series(g["streamflow"].values, index=pd.to_datetime(g["date"].values)).sort_index()
        s = s[~s.index.duplicated(keep="last")].asfreq("D")
        df = pd.DataFrame({"flow": s, "flow_lag1": s.shift(1), "flow_lag7": s.shift(7)})
        df["doy_sin"], df["doy_cos"] = seasonal_features(df.index)
        df["station_id"] = sid
        frames.append(df.dropna())
    out = pd.concat(frames)
    out.index.name = "date"
    return out.reset_index()


def feature_row(station_id, target_date, lag1, lag7, feature_columns) -> pd.DataFrame:
    s, c = seasonal_features([pd.Timestamp(target_date)])
    row = {col: 0.0 for col in feature_columns}
    row.update({"doy_sin": s[0], "doy_cos": c[0], "flow_lag1": lag1, "flow_lag7": lag7})
    site_col = f"site_{station_id}"
    if site_col in row:
        row[site_col] = 1.0
    return pd.DataFrame([row], columns=feature_columns)


# --------------------------------------------------------------------------- climatology
def build_climatology(history: pd.DataFrame) -> pd.DataFrame:
    """Percentiles of the 7-day mean flow, per station and day-of-year (+/- CLIM_WINDOW days)."""
    rows = []
    for sid, g in history.groupby("station_id"):
        s = pd.Series(g["streamflow"].values, index=pd.to_datetime(g["date"].values)).sort_index()
        s = s[~s.index.duplicated(keep="last")].asfreq("D")
        r7 = s.rolling(7, min_periods=4).mean().dropna()
        doy, vals = doy365(r7.index), r7.values
        for d in range(1, 366):
            diff = np.abs(doy - d)
            diff = np.minimum(diff, 365 - diff)
            window = vals[diff <= CLIM_WINDOW]
            if len(window) < 30:
                continue
            q = np.percentile(window, LEVELS)
            rows.append({"station_id": sid, "doy": d, **{f"p{l}": v for l, v in zip(LEVELS, q)}})
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def load_climatology():
    if not CLIM_FILE.exists():
        return None
    return pd.read_csv(CLIM_FILE, dtype={"station_id": str})


def percentile_of(value, quantile_values) -> float:
    """Approximate percentile (1-99) of value given the stored quantiles."""
    return float(np.interp(value, quantile_values, LEVELS, left=1.0, right=99.0))


def drought_status(pct):
    if pct is None:
        return {"code": "UNKNOWN", "label": "No reference data"}
    if pct <= 2:
        return {"code": "D4", "label": "Exceptionally low flow"}
    if pct <= 5:
        return {"code": "D3", "label": "Extremely low flow"}
    if pct <= 10:
        return {"code": "D2", "label": "Severely low flow"}
    if pct <= 20:
        return {"code": "D1", "label": "Moderately low flow"}
    if pct <= 30:
        return {"code": "D0", "label": "Abnormally dry"}
    if pct >= 90:
        return {"code": "HIGH", "label": "High flow"}
    return {"code": "NORMAL", "label": "Normal"}


# --------------------------------------------------------------------------- model
@lru_cache(maxsize=1)
def load_models():
    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    models = {}
    for name in ("q10", "q50", "q90"):
        m = XGBRegressor()
        m.load_model(str(MODEL_DIR / f"xgb_{name}.json"))
        models[name] = m
    return meta, models


def forecast_next_day(station_id, series: pd.Series):
    """Forecast for the day after the last observation. Needs flow(D) and flow(D-6)."""
    loaded = load_models()
    if loaded is None or series.dropna().empty:
        return None
    meta, models = loaded
    s = series.sort_index().asfreq("D")
    last = s.dropna().index.max()
    lag1, lag7 = s.get(last), s.get(last - pd.Timedelta(days=6))
    if pd.isna(lag1) or lag7 is None or pd.isna(lag7):
        return None
    target = last + pd.Timedelta(days=1)
    X = feature_row(station_id, target, lag1, lag7, meta["feature_columns"])
    q10, q50, q90 = sorted(max(0.0, float(models[n].predict(X)[0])) for n in ("q10", "q50", "q90"))
    return {"date": str(target.date()), "q10": q10, "q50": q50, "q90": q90}


# --------------------------------------------------------------------------- station analysis
def analyze_station(station_id: str, obs: pd.DataFrame):
    """Latest flow, 7-day mean, percentile vs seasonal normal, drought status, forecast."""
    rows = obs[obs["station_id"] == station_id].dropna(subset=["streamflow"])
    if rows.empty:
        return None
    s = pd.Series(rows["streamflow"].values, index=pd.to_datetime(rows["date"].values)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    last = s.index.max()
    last_week = s[s.index > last - pd.Timedelta(days=7)]
    flow_7d = float(last_week.mean()) if len(last_week) >= 4 else None

    pct = anomaly = normal = None
    clim = load_climatology()
    if clim is not None and flow_7d is not None:
        ref = clim[(clim["station_id"] == station_id) & (clim["doy"] == min(last.dayofyear, 365))]
        if not ref.empty:
            qvals = [float(ref.iloc[0][f"p{l}"]) for l in LEVELS]
            pct = round(percentile_of(flow_7d, qvals), 1)
            normal = float(ref.iloc[0]["p50"])
            anomaly = round((flow_7d / normal - 1) * 100, 1) if normal > 0 else None

    return {
        "station_id": station_id,
        "date": str(last.date()),
        "streamflow": float(s.iloc[-1]),
        "flow_7d_mean": flow_7d,
        "normal_median_7d": normal,
        "anomaly_pct": anomaly,
        "percentile": pct,
        "status": drought_status(pct),
        "forecast": forecast_next_day(station_id, s),
    }
