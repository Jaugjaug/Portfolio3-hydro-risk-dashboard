# -*- coding: utf-8 -*-
import pandas as pd
from fastapi.testclient import TestClient

import analysis
import main

client = TestClient(main.app)

dates = pd.date_range("2026-10-01", periods=10).strftime("%Y-%m-%d")
FAKE = pd.DataFrame({
    "station_id": ["06892000"] * 10 + ["14166500"],
    "date": list(dates) + ["2026-10-10"],
    "streamflow": [10.0 + i for i in range(10)] + [99.0],
})


def patch(monkeypatch):
    monkeypatch.setattr(main, "load_data", lambda: FAKE)
    monkeypatch.setattr(analysis, "load_climatology", lambda: None)
    monkeypatch.setattr(analysis, "load_models", lambda: None)


def test_status_returns_latest_row(monkeypatch):
    patch(monkeypatch)
    r = client.get("/status/06892000")
    assert r.status_code == 200
    assert r.json() == {"station_id": "06892000", "date": "2026-10-10", "streamflow": 19.0}


def test_unknown_station_returns_404(monkeypatch):
    patch(monkeypatch)
    assert client.get("/status/00000000").status_code == 404
    assert client.get("/analysis/00000000").status_code == 404
    assert client.get("/history/00000000").status_code == 404


def test_analysis_without_reference_data(monkeypatch):
    patch(monkeypatch)
    r = client.get("/analysis/06892000").json()
    assert r["streamflow"] == 19.0
    assert r["flow_7d_mean"] == 16.0          # mean of 13..19
    assert r["percentile"] is None and r["forecast"] is None
    assert r["status"]["code"] == "UNKNOWN"


def test_history_limits_days(monkeypatch):
    patch(monkeypatch)
    assert len(client.get("/history/06892000?days=3").json()) == 3


def test_health():
    assert client.get("/health").json() == {"status": "ok"}
