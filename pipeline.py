"""
pipeline.py
-----------
End-to-end runner for the Alberta electricity demand forecasting pipeline.

Usage:
    python pipeline.py                     # Full run: ingest → features → train → visualize
    python pipeline.py --skip-download     # Use existing data/raw/ CSVs
    python pipeline.py --eda-only          # EDA charts only, no model training
"""

import argparse
import sys
from pathlib import Path

src = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(src))

from data_ingestion import ingest
from feature_engineering import build_feature_matrix
from train import train_xgboost, train_lightgbm, summarize_results, XGB_AVAILABLE, LGB_AVAILABLE
from visualize import (
    plot_demand_overview,
    plot_seasonal_load_profiles,
    plot_demand_heatmap,
    plot_forecast_vs_actual,
    plot_error_distribution,
    plot_feature_importance,
)

PROCESSED = Path(__file__).resolve().parent / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def run(skip_download: bool = False, eda_only: bool = False):
    print("=" * 60)
    print("Alberta Electricity Demand Forecasting Pipeline")
    print("Data source: AESO (Alberta Electric System Operator)")
    print("=" * 60)

    # Step 1: Ingest
    print("\n[1/4] Data Ingestion")
    raw_df = ingest(download=not skip_download)
    raw_df.to_parquet(PROCESSED / "aeso_raw.parquet", index=False)

    # Step 2: EDA charts (pre-feature engineering — raw data)
    print("\n[2/4] Exploratory Visualizations")
    plot_demand_overview(raw_df)
    plot_seasonal_load_profiles(raw_df)
    plot_demand_heatmap(raw_df)

    if eda_only:
        print("\nEDA-only mode — done.")
        return

    # Step 3: Feature engineering
    print("\n[3/4] Feature Engineering")
    feature_df = build_feature_matrix(raw_df)
    feature_df.to_parquet(PROCESSED / "aeso_features.parquet", index=False)

    # Step 4: Train models
    print("\n[4/4] Model Training (Walk-Forward Validation)")
    results = []

    if XGB_AVAILABLE:
        print("\n  XGBoost:")
        results.append(train_xgboost(feature_df, n_folds=5))

    if LGB_AVAILABLE:
        print("\n  LightGBM:")
        results.append(train_lightgbm(feature_df, n_folds=5))

    if not results:
        print("No models trained — install xgboost or lightgbm")
        return

    # Summaries and charts
    summary = summarize_results(results)
    summary.to_csv(
        Path(__file__).resolve().parent / "outputs" / "model_comparison.csv",
        index=False
    )

    plot_error_distribution(results)

    for result in results:
        plot_forecast_vs_actual(
            result.actuals,
            result.predictions,
            model_name=result.model_name,
        )

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Charts  → outputs/figures/")
    print(f"  Models  → models/")
    print(f"  Results → outputs/model_comparison.csv")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AESO Demand Forecasting Pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download, use existing CSVs in data/raw/")
    parser.add_argument("--eda-only", action="store_true",
                        help="Run EDA visualizations only, skip model training")
    args = parser.parse_args()

    run(skip_download=args.skip_download, eda_only=args.eda_only)
