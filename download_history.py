# -*- coding: utf-8 -*-
"""One-off: download long daily streamflow history (USGS) for the 20 stations.
Output: data/raw/history.csv.gz (used to build the climatology and train the model)."""

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dataretrieval import waterdata

from config import STATIONS

START = "2000-01-01"
OUT = Path("data/raw/history.csv.gz")


def fetch(site_id, start, end, retries=3):
    for attempt in range(1, retries + 1):
        try:
            df, _ = waterdata.get_daily(
                monitoring_location_id=f"USGS-{site_id}",
                parameter_code="00060",
                time=[start, end],
            )
            df = df[df["statistic_id"] == "00003"]           # daily mean
            return pd.DataFrame({
                "station_id": site_id,
                "date": pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d"),
                "streamflow": pd.to_numeric(df["value"], errors="coerce"),
            }).dropna()
        except Exception as exc:
            print(f"[WARN] {site_id} attempt {attempt}/{retries}: {exc}")
            time.sleep(3 * attempt)
    return pd.DataFrame(columns=["station_id", "date", "streamflow"])


def main():
    end = str(date.today() - timedelta(days=1))
    frames = []
    for i, site in enumerate(STATIONS, start=1):
        df = fetch(site, START, end)
        print(f"[{i}/{len(STATIONS)}] {site}: {len(df)} days")
        frames.append(df)
        time.sleep(1.5)              # be gentle with the API
    history = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(OUT, index=False, compression="gzip")
    print(f"Saved {len(history)} rows to {OUT}")


if __name__ == "__main__":
    main()
