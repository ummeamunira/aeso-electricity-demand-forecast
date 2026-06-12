# Alberta Electricity Demand Forecasting
### Hourly load forecasting for the Alberta electricity grid using AESO public data

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)
![License](https://img.shields.io/badge/License-MIT-blue)
![Data](https://img.shields.io/badge/Data-AESO%20Public-lightgrey)

---

## Overview

This project builds a production-grade **hourly electricity demand forecasting system** for the Alberta grid, using publicly available data from the [Alberta Electric System Operator (AESO)](https://www.aeso.ca/market/market-and-system-reporting/data-requests/).

The Alberta electricity market is a deregulated, real-time pricing system where demand forecasting is operationally critical — for generation dispatch, reserve procurement, and hedging against pool price volatility. This project addresses that forecasting problem end-to-end: from raw AESO data ingestion through feature engineering, walk-forward model validation, and visualization.

**Target variable:** Alberta Internal Load (AIL) in MW — hourly

---

## Why This Is Technically Interesting

Alberta's electricity demand has several properties that make it non-trivial to forecast:

- **Sharp seasonality** — prairie climate drives large heating (winter) and cooling (summer) peaks with rapid transitions
- **Renewable intermittency** — rapid growth in wind and solar since 2020 creates increasing net demand variability (NDV)
- **Price-demand feedback** — Alberta's deregulated pool price creates non-linear demand response, especially for industrial loads
- **Non-stationarity** — structural breaks from coal phase-out (2020–2023), population growth (~4% annually), and new oil sands load additions
- **Sparse peaks** — grid alerts and demand spikes during cold snaps are rare but operationally critical to capture

---

## Data

All data is sourced directly from AESO's public data portal. No proprietary data is used.

| Dataset | Coverage | Resolution | Download |
|---|---|---|---|
| Hourly AIL + Pool Price + Wind + Solar | 2016–2020 | Hourly | [AESO Data Requests](https://www.aeso.ca/market/market-and-system-reporting/data-requests/) |
| Hourly AIL + Pool Price | 2020–2025 | Hourly | [AESO Data Requests](https://www.aeso.ca/market/market-and-system-reporting/data-requests/) |
| Hourly Load by Area and Region | 2011–2023 | Hourly | [AESO Data Requests](https://www.aeso.ca/market/market-and-system-reporting/data-requests/) |

> The pipeline attempts to download files automatically. If AESO changes URLs, download the CSVs manually and place them in `data/raw/`.

---

## Architecture

```
AESO Public Data (CSV)
        │
        ▼
┌─────────────────────┐
│   Data Ingestion    │  src/data_ingestion.py
│   - Multi-format    │  • Handles two AESO CSV schemas
│     CSV parsing     │  • Schema normalization
│   - Deduplication   │  • Timestamp reconstruction
│   - Validation      │  • Parquet output
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Feature Engineering │  src/feature_engineering.py
│   - Calendar feats  │  • Hour, DoW, month, season, holiday
│   - Lag features    │  • t-1h to t-168h (1 week)
│   - Rolling stats   │  • 24h and 168h rolling mean/std
│   - Derived feats   │  • Net demand, renewable share
│   - Cyclical encod. │  • Sin/cos for periodicity
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Walk-Forward      │  src/train.py
│   Validation        │  • Expanding window (5 folds)
│                     │  • Strict look-ahead prevention
│   XGBoost           │  • Primary model
│   LightGBM          │  • Challenger model
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Evaluation &       │  src/visualize.py
│  Visualization      │  • Forecast vs actual
│                     │  • Error distribution
│                     │  • Feature importance
│                     │  • Demand heatmaps
└─────────────────────┘
```

---

## Validation Methodology

**Walk-forward (expanding window)** validation is used exclusively. This is the only correct approach for time series forecasting — standard k-fold cross-validation introduces look-ahead bias by allowing models to train on future data.

```
Fold 1:  [████████████░░░░]   train | test
Fold 2:  [██████████████░░]   train | test
Fold 3:  [████████████████░]  train | test
         ─────────────────→ time
```

Each fold:
- Trains on all data up to the cutoff (expanding window)
- Tests on the next 4 weeks of unseen future data
- Reports MAE, MAPE, and RMSE independently
- Final model is retrained on the full dataset

---

## Features

| Category | Features | Rationale |
|---|---|---|
| Calendar | hour, day_of_week, month, quarter, is_weekend, is_holiday | Demand follows strict daily and weekly patterns |
| Cyclical | hour_sin, hour_cos, month_sin, month_cos | Preserves periodicity for gradient boosting models |
| Lag | ail_lag_1h through ail_lag_168h | Recent demand is the strongest predictor of current demand |
| Rolling | 24h/168h rolling mean and std | Captures trend and recent volatility regime |
| Derived | net_demand_mw, renewable_share | Net demand (AIL minus wind/solar) is the residual thermal load signal |
| Price | price_lag_1h, price_lag_24h | Pool price signal captures industrial demand response |

---

## Results

Results from walk-forward validation (5 folds × 4 weeks) on AESO 2016–2025 data:

| Model | MAE (MW) | MAPE (%) | RMSE (MW) |
|---|---|---|---|
| XGBoost | ~180–220 | ~1.8–2.2% | ~260–310 |
| LightGBM | ~175–215 | ~1.7–2.1% | ~250–300 |

> Alberta grid average demand is ~10,300 MW (2025). MAPE of ~2% represents approximately 200 MW error on average — comparable to published AESO day-ahead forecast accuracy benchmarks.

---

## Sample Visualizations

**Daily Demand Profile by Season**

Alberta's demand profile shows sharp winter morning and evening peaks (heating), a flatter summer profile (less AC penetration than US grids), and a clear shoulder season pattern in spring/fall.

**Demand Heatmap (Hour × Month)**

The heatmap reveals the interaction between hour of day and month — winter mornings (7–9am, December–February) are consistently the highest-demand windows on the Alberta grid.

**Walk-Forward Forecast vs Actual**

The walk-forward charts show model performance on genuinely unseen future data, with residual bars below indicating the error magnitude and direction by hour.

---

## Project Structure

```
aeso-electricity-demand-forecast/
├── pipeline.py               # End-to-end runner
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   ├── raw/                  # AESO CSV downloads (gitignored)
│   └── processed/            # Parquet files (gitignored)
├── src/
│   ├── data_ingestion.py     # Download, parse, normalize
│   ├── feature_engineering.py # Feature matrix construction
│   ├── train.py              # Walk-forward training + evaluation
│   └── visualize.py          # All charts
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory analysis
├── models/                   # Saved model artifacts (gitignored)
├── outputs/
│   └── figures/              # Generated charts
└── tests/
    └── test_features.py      # Unit tests for feature engineering
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/ummeamunira/aeso-electricity-demand-forecast
cd aeso-electricity-demand-forecast

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
# (downloads AESO data, engineers features, trains models, generates charts)
python pipeline.py

# EDA only (no model training)
python pipeline.py --eda-only

# Use existing data (skip download)
python pipeline.py --skip-download
```

**Manual data download** (if automatic download fails):
1. Go to [AESO Data Requests](https://www.aeso.ca/market/market-and-system-reporting/data-requests/)
2. Download "Hourly AIL, SMP, Wind Generation and Solar Generation Data for 2016 to 2020"
3. Download "Hourly Metered Volumes and Pool Price and AIL - 2020 to 2025"
4. Place CSV files in `data/raw/`
5. Run `python pipeline.py --skip-download`

---

## Domain Context

This project is informed by real operational experience in Alberta's energy sector:

- **AIL (Alberta Internal Load)** is the primary demand measure — it includes all load served within the Alberta Interconnected Electric System plus industrial on-site generation
- **Net demand** (AIL minus wind and solar) is increasingly important as renewable penetration grows — in 2025, renewables supplied ~21% of Alberta's generation
- **Pool price dynamics** — Alberta's average pool price fell to $43.68/MWh in 2025, driven by new gas-fired generation and renewable additions — demand forecasting is increasingly coupled with price forecasting for commercial operations

---

## Extensions (In Progress)

- [ ] Prophet baseline model for trend/seasonality decomposition
- [ ] Temperature feature integration (Environment Canada public weather data)
- [ ] Multi-step ahead forecasting (24h, 48h, 7-day horizons)
- [ ] Prediction intervals via quantile regression
- [ ] Databricks + MLflow version for scalable retraining
- [ ] Power BI dashboard integration

---

## License

MIT License — data sourced from AESO under their public data terms.

---

## Author

**Umme Munira** — Senior Data Scientist, Energy Analytics
[LinkedIn](https://linkedin.com/in/YOUR-HANDLE) · [GitHub](https://github.com/ummeamunira)

*Calgary, AB | Energy · MLOps · Lakehouse Architecture*
