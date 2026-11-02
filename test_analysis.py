# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import analysis
import train_model


def synthetic_history(years=6, stations=("01111111", "02222222")):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=365 * years)
    frames = []
    for i, sid in enumerate(stations):
        base = 50 * (i + 1) + 40 * (i + 1) * np.sin(2 * np.pi * dates.dayofyear / 365)
        flow = np.clip(base + rng.normal(0, 3, len(dates)), 1, None)
        frames.append(pd.DataFrame({"station_id": sid, "date": dates.strftime("%Y-%m-%d"), "streamflow": flow}))
    return pd.concat(frames, ignore_index=True)


def test_drought_status_levels():
    assert analysis.drought_status(1)["code"] == "D4"
    assert analysis.drought_status(8)["code"] == "D2"
    assert analysis.drought_status(50)["code"] == "NORMAL"
    assert analysis.drought_status(95)["code"] == "HIGH"
    assert analysis.drought_status(None)["code"] == "UNKNOWN"


def test_percentile_is_monotonic():
    q = [1, 2, 3, 5, 8, 10, 15, 20, 25]
    assert analysis.percentile_of(0, q) < analysis.percentile_of(10, q) < analysis.percentile_of(100, q)


def test_lags_use_calendar_days():
    h = pd.DataFrame({"station_id": "x", "date": ["2026-01-01", "2026-01-02", "2026-01-04"],
                      "streamflow": [1.0, 2.0, 4.0]})
    # the gap on Jan 3 means no row can have a valid lag7 -> nothing is built from misaligned rows
    assert analysis.build_training_frame(h).empty


def test_train_and_forecast_end_to_end(tmp_path, monkeypatch):
    hist = synthetic_history()
    meta = train_model.train(hist, tmp_path / "models", tmp_path / "clim.csv", n_estimators=30)
    assert meta["metrics"]["nse_model"] > 0.5

    monkeypatch.setattr(analysis, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(analysis, "CLIM_FILE", tmp_path / "clim.csv")
    analysis.load_models.cache_clear()
    analysis.load_climatology.cache_clear()

    recent = hist[hist["station_id"] == "01111111"].tail(10)
    res = analysis.analyze_station("01111111", recent)
    assert res["percentile"] is not None
    assert res["forecast"] is not None
    assert res["forecast"]["q10"] <= res["forecast"]["q50"] <= res["forecast"]["q90"]

    # a very low flow must be flagged as dry
    low = recent.copy()
    low["streamflow"] = 0.1
    assert analysis.analyze_station("01111111", low)["status"]["code"] == "D4"
    analysis.load_models.cache_clear()
    analysis.load_climatology.cache_clear()
