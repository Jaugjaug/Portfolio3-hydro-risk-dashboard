# -*- coding: utf-8 -*-
"""Daily pipeline: fetch USGS daily streamflow and append it to data/observations.csv."""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dataretrieval import waterdata

from config import STATIONS

DATA_FILE = Path("data/observations.csv")
COLUMNS = ["station_id", "date", "streamflow"]
PARAM_CD = "00060"      # discharge, cubic feet per second
STAT_MEAN = "00003"     # USGS statistic code for daily mean
LOOKBACK_DAYS = 14      # re-fetch two weeks so a skipped run never leaves a gap


def fetch_station(site, start, end):
    df, _ = waterdata.get_daily(
        monitoring_location_id=f"USGS-{site}",
        parameter_code=PARAM_CD,
        time=f"{start}/{end}",
    )
    if df is None or len(df) == 0:
        return []

    # Keep daily means only (the service can also return min/max series)
    if "statistic_id" in df.columns:
        df = df[df["statistic_id"].astype(str) == STAT_MEAN]

    missing = {"time", "value"} - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected columns, missing {missing}: {list(df.columns)}")

    dates = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
    values = pd.to_numeric(df["value"], errors="coerce")
    return [
        {"station_id": site, "date": d, "streamflow": float(v)}
        for d, v in zip(dates, values)
        if pd.notna(v)
    ]


def main():
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=LOOKBACK_DAYS - 1)

    new_rows, failures = [], 0
    for site in STATIONS:
        try:
            new_rows.extend(fetch_station(site, start, end))
        except Exception as exc:
            failures += 1
            print(f"[WARN] station {site}: {exc}", file=sys.stderr)

    if failures == len(STATIONS):
        print("[ERROR] All stations failed, nothing written.", file=sys.stderr)
        sys.exit(1)

    new_df = pd.DataFrame(new_rows, columns=COLUMNS)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists():
        existing = pd.read_csv(DATA_FILE, dtype={"station_id": str})
    else:
        existing = pd.DataFrame(columns=COLUMNS)

    combined = (
        pd.concat([existing, new_df], ignore_index=True)
        .drop_duplicates(subset=["station_id", "date"], keep="last")
        .sort_values(["station_id", "date"])
    )
    combined.to_csv(DATA_FILE, index=False)
    print(f"{len(new_df)} rows fetched, {len(combined)} rows in {DATA_FILE}, {failures} station(s) failed.")


if __name__ == "__main__":
    main()