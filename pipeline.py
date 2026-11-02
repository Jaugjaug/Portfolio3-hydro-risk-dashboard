# -*- coding: utf-8 -*-
"""Daily pipeline: fetch USGS daily streamflow and append it to data/observations.csv."""

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dataretrieval import nwis

from config import STATIONS

DATA_FILE = Path("data/observations.csv")
COLUMNS = ["station_id", "date", "streamflow"]
PARAM_CD = "00060"      # discharge, cubic feet per second
LOOKBACK_DAYS = 14      # re-fetch two weeks so a skipped run never leaves a gap


def find_flow_column(df):
    """Return the daily-mean discharge column (e.g. '00060_Mean'), ignoring '*_cd' flags."""
    candidates = [c for c in df.columns if str(c).startswith(PARAM_CD) and not str(c).endswith("_cd")]
    return candidates[0] if candidates else None


def fetch_station(site, start, end):
    df, _ = nwis.get_dv(sites=site, start=str(start), end=str(end), parameterCd=PARAM_CD)
    if df.empty:
        return []
    col = find_flow_column(df)
    if col is None:
        raise ValueError(f"No discharge column in {list(df.columns)}")
    idx = df.index.get_level_values("datetime") if "datetime" in df.index.names else df.index
    dates = pd.to_datetime(idx).strftime("%Y-%m-%d")
    return [
        {"station_id": site, "date": d, "streamflow": float(v)}
        for d, v in zip(dates, df[col])
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
