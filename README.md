# Hydro-climatic Risk Dashboard

[![Tests](https://github.com/Jaugjaug/Portfolio3-hydro-risk-dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/Jaugjaug/Portfolio3-hydro-risk-dashboard/actions/workflows/tests.yml)
[![Update data](https://github.com/Jaugjaug/Portfolio3-hydro-risk-dashboard/actions/workflows/update_data.yml/badge.svg)](https://github.com/Jaugjaug/Portfolio3-hydro-risk-dashboard/actions/workflows/update_data.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Automated monitoring of daily streamflow for 20 US river basins: seasonal anomaly, drought status and a
1-day-ahead forecast, served by an API and displayed in a dashboard. The data refreshes every day with no
manual action.

**[Live dashboard](https://portfolio3-hydro-risk-api.streamlit.app/)** | **[API documentation](https://portfolio3-hydro-risk-api.onrender.com/docs)**

> The free hosting tiers go to sleep when idle: the first request can take 30-60 seconds.

<!-- Add a demo once docs/demo.gif exists: ![Dashboard demo](docs/demo.gif) -->

## Features

- **Daily automated pipeline**: a scheduled GitHub Actions workflow fetches new USGS data and commits it to the repository.
- **Seasonal anomaly and drought status**: the 7-day mean flow is compared with the history of the same station and season (2000 onwards). Its percentile gives a status from D0 to D4 (low flow) or HIGH (high flow).
- **1-day-ahead forecast**: XGBoost model with an 80% prediction interval (10% / 90% quantile models).
- **REST API** (FastAPI) with interactive documentation.
- **Dashboard** (Streamlit) with a per-basin view and a map.
- **Automated tests** on synthetic data (no Internet required), run by CI on every push.

## Architecture

```
GitHub Actions (daily, 14:00 UTC)
        |  commits
        v
data/observations.csv --> FastAPI (main.py + analysis.py) --> Streamlit (dashboard.py)
                                  ^
one-off scripts --> data/climatology.csv + models/*.json
```

There is no database: the CSV committed in Git is the data store. The API reads it through
`raw.githubusercontent.com` (cached 15 minutes) and the dashboard queries the API (cached 10 minutes).

| File | Role |
|---|---|
| `pipeline.py` | Daily fetch of USGS data (7-day lookback so a skipped run never leaves a gap) |
| `main.py` | FastAPI routes |
| `analysis.py` | Seasonal normal, percentile, drought status, forecast |
| `dashboard.py` | Streamlit interface |
| `config.py` | Station list and coordinates |
| `download_history.py` | One-off download of the USGS history (2000 onwards) |
| `train_model.py` | One-off training of the climatology and forecast models |
| `test_api.py`, `test_analysis.py` | Tests |
| `.github/workflows/` | `update_data.yml` (daily data update), `tests.yml` (CI) |

## API endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Service health check |
| `GET /status/{station_id}` | Latest observation for a station |
| `GET /analysis/{station_id}` | Anomaly, drought status and 1-day forecast |
| `GET /history/{station_id}?days=90` | Recent daily flow |
| `GET /overview` | Status of all stations |

Station IDs are 8-character USGS identifiers with leading zeros (for example `01134500`).
Flows are returned in cubic feet per second (cfs); the dashboard converts them to m3/s for display only.

## Model performance

Holdout metrics, read from `models/meta.json`:

| Metric | Value |
|---|---|
| NSE, XGBoost model | _to fill_ |
| NSE, persistence baseline (tomorrow = today) | _to fill_ |
| Coverage of the 80% interval | _to fill_ |

Persistence is a very strong baseline for 1-day-ahead streamflow, so the model should be judged against it,
not against zero.

## Getting started

### Prerequisites

- Python 3.12
- Internet access for the USGS data

### Installation

```bash
git clone https://github.com/Jaugjaug/Portfolio3-hydro-risk-dashboard.git
cd Portfolio3-hydro-risk-dashboard
python -m venv venv-dashboard
```

Activate the environment:

```bash
# Mac/Linux
source venv-dashboard/bin/activate
```

```powershell
# Windows PowerShell
venv-dashboard\Scripts\Activate.ps1
```

```bat
:: Windows cmd
venv-dashboard\Scripts\activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### One-off preparation

Only needed to rebuild the model and climatology (the repository already contains them):

```bash
python download_history.py   # daily flow since 2000 -> data/raw/history.csv.gz (a few minutes)
python train_model.py        # -> data/climatology.csv, models/*.json, prints holdout metrics
python pipeline.py           # -> data/observations.csv (recent days)
```

`data/raw/` is git-ignored. Do not open or save the CSV files with Excel: it strips the leading zeros of the station IDs.

### Run locally

Terminal 1, API (run from the project folder):

```bash
# Mac/Linux
export DATA_URL=data/observations.csv
uvicorn main:app --reload
```

```powershell
# Windows PowerShell
$env:DATA_URL="data/observations.csv"
uvicorn main:app --reload
```

```bat
:: Windows cmd (never put a comment on the same line as "set")
set DATA_URL=data/observations.csv
uvicorn main:app --reload
```

The API is then available at http://127.0.0.1:8000/docs.

Terminal 2, dashboard:

```bash
streamlit run dashboard.py
```

Open http://localhost:8501.

### Configuration

| Variable | Used by | Default |
|---|---|---|
| `DATA_URL` | API | raw GitHub URL of `data/observations.csv` (can be a local path) |
| `API_URL` | Dashboard | `http://127.0.0.1:8000` |

## Tests

```bash
pytest
```

The tests use synthetic data and do not need Internet access.

## Deployment

| Component | Platform | Settings |
|---|---|---|
| Data pipeline | GitHub Actions | `.github/workflows/update_data.yml`, daily at 14:00 UTC, can also be triggered manually. Requires *Settings > Actions > General > Workflow permissions: Read and write*. |
| API | Render (free tier) | Build: `pip install -r requirements.txt`. Start: `uvicorn main:app --host 0.0.0.0 --port 10000`. Do not set `DATA_URL`. |
| Dashboard | Streamlit Community Cloud | Main file `dashboard.py`. Secret: `API_URL = "https://portfolio3-hydro-risk-api.onrender.com"` (no trailing slash). |

## Limitations

- Values are **daily mean flows up to yesterday**, published with a delay by USGS. This is not real-time monitoring.
- Recent USGS values are provisional and may be revised.
- The forecast covers one day ahead only.
- The drought status is a statistical indicator based on flow percentiles, not an official drought classification.
- The model and seasonal normal are trained once and should be retrained once or twice a year.
- The free hosting tiers sleep when idle, and scheduled GitHub workflows can be disabled after about 60 days without repository activity (the daily data commits count as activity).

## Data source

Streamflow data: [USGS National Water Information System](https://waterdata.usgs.gov/), retrieved with the
[`dataretrieval`](https://github.com/DOI-USGS/dataretrieval-python) package. USGS data are public domain.

## Related projects

- [Portfolio 1: Flow Rate Prediction Using Machine Learning](https://github.com/Jaugjaug/Portfolio1_Flow_Rate_Prediction_Using_Machine_Learning): features and forecasting approach (seasonality, lagged flows, station encoding).
- [Portfolio 2: Sierra Nevada SWE Forecasting](https://github.com/Jaugjaug/portfolio2-sierra-nevada-swe-forecasting): seasonal normal and quantile-regression intervals.

## License

Released under the MIT License. Add a `LICENSE` file at the root of the repository (GitHub can generate it: *Add file > Create new file > name it `LICENSE` > Choose a license template*).
