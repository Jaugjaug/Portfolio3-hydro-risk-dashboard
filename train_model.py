# -*- coding: utf-8 -*-
"""One-off: build the climatology and train the 1-day-ahead XGBoost models (q10 / q50 / q90).
Same features as portfolio 1 (seasonality + lag1 + lag7 + station one-hot)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import analysis

HISTORY = Path("data/raw/history.csv.gz")


def nse(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    return float(1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())


def make_model(alpha=None, n_estimators=300):
    kwargs = dict(n_estimators=n_estimators, learning_rate=0.05, random_state=42, n_jobs=-1)
    if alpha is not None:
        kwargs.update(objective="reg:quantileerror", quantile_alpha=alpha)
    return XGBRegressor(**kwargs)


def train(history: pd.DataFrame, model_dir: Path, clim_file: Path, n_estimators=300, holdout_years=3):
    model_dir.mkdir(parents=True, exist_ok=True)
    clim_file.parent.mkdir(parents=True, exist_ok=True)

    clim = analysis.build_climatology(history)
    clim.to_csv(clim_file, index=False)

    frame = analysis.build_training_frame(history)
    X = pd.concat([frame[analysis.BASE_FEATURES],
                   pd.get_dummies(frame["station_id"], prefix="site", dtype=float)], axis=1)
    y = frame["flow"]
    columns = list(X.columns)

    # 1) honest evaluation on the last years (never seen in training)
    cutoff = frame["date"].max() - pd.DateOffset(years=holdout_years)
    test = frame["date"] > cutoff
    metrics = {"holdout_start": str(cutoff.date()), "n_test": int(test.sum())}
    if test.any() and (~test).any():
        m50 = make_model(None, n_estimators).fit(X[~test], y[~test])
        m10 = make_model(0.1, n_estimators).fit(X[~test], y[~test])
        m90 = make_model(0.9, n_estimators).fit(X[~test], y[~test])
        p50, p10, p90 = m50.predict(X[test]), m10.predict(X[test]), m90.predict(X[test])
        lo, hi = np.minimum(p10, p90), np.maximum(p10, p90)
        metrics.update({
            "nse_model": nse(y[test], p50),
            "nse_persistence": nse(y[test], frame.loc[test, "flow_lag1"]),
            "rmse_model": float(np.sqrt(((y[test] - p50) ** 2).mean())),
            "rmse_persistence": float(np.sqrt(((y[test] - frame.loc[test, 'flow_lag1']) ** 2).mean())),
            "coverage_q10_q90": float(((y[test] >= lo) & (y[test] <= hi)).mean()),
        })

    # 2) final models on all the data
    for name, alpha in (("q50", None), ("q10", 0.1), ("q90", 0.9)):
        make_model(alpha, n_estimators).fit(X, y).save_model(str(model_dir / f"xgb_{name}.json"))

    meta = {
        "feature_columns": columns,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": int(len(frame)),
        "metrics": metrics,
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    hist = pd.read_csv(HISTORY, dtype={"station_id": str})
    meta = train(hist, Path("models"), analysis.CLIM_FILE)
    print(json.dumps(meta["metrics"], indent=2))
