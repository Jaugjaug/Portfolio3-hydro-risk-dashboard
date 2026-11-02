# -*- coding: utf-8 -*-
"""Streamlit dashboard querying the FastAPI service."""

import os

import pandas as pd
import requests
import streamlit as st

from config import STATIONS

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
CFS_TO_M3S = 0.028316846592   # 1 cubic foot per second in m3/s
COLORS = {"D4": "#730000", "D3": "#E60000", "D2": "#FFAA00", "D1": "#FCD37F",
          "D0": "#FFFF00", "NORMAL": "#2E9E4F", "HIGH": "#1F77B4", "UNKNOWN": "#999999"}


def api_get(path: str):
    # Long timeout: free hosting can take 30-60 s to wake up
    r = requests.get(f"{API_URL}{path}", timeout=90)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600, show_spinner="Fetching data (the API may be waking up)...")
def fetch(path: str):
    return api_get(path)


def m3s(cfs: float) -> str:
    """Format a flow given in cfs (API unit) as m3/s."""
    return f"{cfs * CFS_TO_M3S:,.3f} m³/s"


def show_alert(status: dict):
    msg = f"{status['code']} - {status['label']}"
    if status["code"] in ("D2", "D3", "D4"):
        st.error(msg)
    elif status["code"] in ("D0", "D1"):
        st.warning(msg)
    elif status["code"] == "HIGH":
        st.info(msg)
    elif status["code"] == "NORMAL":
        st.success(msg)
    else:
        st.write(msg)


st.title("Hydro-climatic Risk Monitoring")
tab_station, tab_map = st.tabs(["Basin", "Map"])

with tab_station:
    station_id = st.selectbox("Basin", list(STATIONS))
    try:
        data = fetch(f"/analysis/{station_id}")
        history = fetch(f"/history/{station_id}?days=90")
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            st.warning("No data available yet for this basin.")
        else:
            st.error("The API returned an error. Please try again later.")
    except requests.RequestException:
        st.error("Cannot reach the API. Please try again in a minute.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest daily mean flow", m3s(data["streamflow"]))
        if data["flow_7d_mean"] is not None:
            c2.metric("7-day mean", m3s(data["flow_7d_mean"]),
                      None if data["anomaly_pct"] is None else f"{data['anomaly_pct']:+.0f}% vs normal")
        if data["percentile"] is not None:
            c3.metric("Percentile (season)", f"{data['percentile']:.0f}")
        show_alert(data["status"])
        st.caption(f"Last observation: {data['date']} (daily means, published with a delay by USGS)")

        fc = data.get("forecast")
        if fc:
            st.subheader(f"Forecast for {fc['date']}")
            st.write(f"Median **{m3s(fc['q50'])}**  (80% interval: {m3s(fc['q10'])} - {m3s(fc['q90'])})")

        st.subheader("Last 90 days")
        hist = pd.DataFrame(history)
        hist["date"] = pd.to_datetime(hist["date"])
        hist["Streamflow (m³/s)"] = hist["streamflow"] * CFS_TO_M3S
        st.line_chart(hist.set_index("date")["Streamflow (m³/s)"])

with tab_map:
    try:
        overview = pd.DataFrame(fetch("/overview"))
    except requests.RequestException:
        st.error("Cannot load the overview.")
    else:
        overview["color"] = overview["status"].map(lambda s: COLORS.get(s["code"], "#999999"))
        overview["status_label"] = overview["status"].map(lambda s: s["label"])
        st.map(overview, latitude="lat", longitude="lon", color="color", size=40000)
        overview["streamflow_m3s"] = (overview["streamflow"] * CFS_TO_M3S).round(3)
        st.dataframe(
            overview[["station_id", "date", "streamflow_m3s", "percentile", "anomaly_pct", "status_label"]],
            hide_index=True,
        )