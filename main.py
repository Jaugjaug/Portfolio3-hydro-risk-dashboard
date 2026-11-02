# -*- coding: utf-8 -*-
"""FastAPI service: latest observation, seasonal analysis, drought status and forecast."""

import os
import time

import pandas as pd
from fastapi import FastAPI, HTTPException

import analysis
from config import STATIONS

app = FastAPI(title="Hydro-climatic Risk API")

# Can be overridden (e.g. a local CSV path) with the DATA_URL environment variable
DATA_URL = os.getenv(
    "DATA_URL",
    "https://raw.githubusercontent.com/Jaugjaug/Portfolio3-hydro-risk-dashboard/main/data/observations.csv",
)
CACHE_TTL = 15 * 60  # seconds
_cache = {"df": None, "ts": 0.0}


def load_data() -> pd.DataFrame:
    now = time.time()
    if _cache["df"] is None or now - _cache["ts"] > CACHE_TTL:
        try:
            df = pd.read_csv(DATA_URL, dtype={"station_id": str})
        except Exception as exc:
            if _cache["df"] is not None:
                return _cache["df"]
            raise HTTPException(
                status_code=503,
                detail=f"Data source unavailable: {DATA_URL} ({type(exc).__name__}: {exc})",
            )
        _cache["df"], _cache["ts"] = df, now
    return _cache["df"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status/{station_id}")
def get_status(station_id: str):
    df = load_data()
    rows = df[df["station_id"] == station_id].dropna(subset=["streamflow"])
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No data for station {station_id}")
    row = rows.sort_values("date").iloc[-1]
    return {"station_id": station_id, "date": str(row["date"]), "streamflow": float(row["streamflow"])}


@app.get("/analysis/{station_id}")
def get_analysis(station_id: str):
    result = analysis.analyze_station(station_id, load_data())
    if result is None:
        raise HTTPException(status_code=404, detail=f"No data for station {station_id}")
    return result


@app.get("/history/{station_id}")
def get_history(station_id: str, days: int = 90):
    df = load_data()
    rows = df[df["station_id"] == station_id].dropna(subset=["streamflow"]).sort_values("date")
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No data for station {station_id}")
    rows = rows.tail(max(1, min(days, 3650)))
    return [{"date": str(r.date), "streamflow": float(r.streamflow)} for r in rows.itertuples()]


@app.get("/overview")
def get_overview():
    df = load_data()
    out = []
    for sid, (lat, lon) in STATIONS.items():
        result = analysis.analyze_station(sid, df)
        if result is not None:
            out.append({**result, "lat": lat, "lon": lon})
    return out
