#!/usr/bin/env python3
"""Build Evidently reports for the Bike Sharing exam in one single run.

Outputs (all in the same project folder):
  - 02_validation.html/json           (validation train vs test - Jan 1–28)
  - 03_prod_january.html/json         (prod model performance on January full)
  - 04_week{1,2,3}_regression.html/json (monitoring weeks of February)
  - 05_target_drift_<worstweek>.html/json (target drift on worst week)
  - 06_data_drift_week3_numeric.html/json (data drift on last week, numeric only)
  - 00_summary.json                   (MAE by week + worst week)

Run once; no manual steps required after.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import os
import warnings
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn import ensemble, model_selection

from evidently.metric_preset import (
    RegressionPreset,
    DataDriftPreset,
    TargetDriftPreset,
)
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report

# Silence noisy warnings as in the statement examples
warnings.filterwarnings("ignore")

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
)

TARGET = "cnt"
PREDICTION = "prediction"
NUMERICAL_FEATURES = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "mnth",
    "hr",
    "weekday",
]
CATEGORICAL_FEATURES = ["season", "holiday", "workingday"]

WEEKS = {
    "week1": ("2011-01-29 00:00:00", "2011-02-07 23:00:00"),
    "week2": ("2011-02-07 00:00:00", "2011-02-14 23:00:00"),
    "week3": ("2011-02-15 00:00:00", "2011-02-21 23:00:00"),
}


# -----------------------------
# Internal helper functions
# -----------------------------
def _fetch_data() -> pd.DataFrame:
    """Download and load hour.csv; set hourly datetime index."""
    resp = requests.get(DATA_URL, verify=False, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as arc:
        df = pd.read_csv(
            arc.open("hour.csv"), header=0, sep=",", parse_dates=["dteday"]
        )
    # Build proper hourly datetime index
    df.index = df.apply(
        lambda r: dt.datetime.combine(r.dteday.date(), dt.time(int(r["hr"]))),
        axis=1,
    )  # type: ignore
    return df


def _build_column_mapping(
    numerical: list[str], categorical: list[str], target: str, prediction: str
) -> ColumnMapping:
    cm = ColumnMapping()
    cm.target = target
    cm.prediction = prediction
    cm.numerical_features = numerical
    cm.categorical_features = categorical
    return cm


def _save_report(rep: Report, out_html: str, out_json: str | None = None) -> None:
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    rep.save_html(out_html)
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rep.as_dict(), f, ensure_ascii=False, indent=2)


def _regression_report(
    reference_df: pd.DataFrame | None,
    current_df: pd.DataFrame,
    column_mapping: ColumnMapping,
    out_html: str,
    out_json: str | None = None,
) -> Report:
    rep = Report(metrics=[RegressionPreset()])
    rep.run(
        reference_data=reference_df.sort_index() if reference_df is not None else None,
        current_data=current_df.sort_index(),
        column_mapping=column_mapping,
    )
    _save_report(rep, out_html, out_json)
    return rep


def _target_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target: str,
    out_html: str,
    out_json: str | None = None,
) -> Report:
    cm = ColumnMapping()
    cm.target = target
    rep = Report(metrics=[TargetDriftPreset()])
    rep.run(
        reference_data=reference_df[[target]].sort_index(),
        current_data=current_df[[target]].sort_index(),
        column_mapping=cm,
    )
    _save_report(rep, out_html, out_json)
    return rep


def _data_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numerical_features: list[str],
    out_html: str,
    out_json: str | None = None,
) -> Report:
    cm = ColumnMapping()
    cm.numerical_features = numerical_features
    rep = Report(metrics=[DataDriftPreset()])
    rep.run(
        reference_data=reference_df[numerical_features].sort_index(),
        current_data=current_df[numerical_features].sort_index(),
        column_mapping=cm,
    )
    _save_report(rep, out_html, out_json)
    return rep


def _train_validation_model(
    jan_ref: pd.DataFrame,
    n_estimators: int,
    random_state: int,
) -> tuple[ensemble.RandomForestRegressor, pd.DataFrame, pd.DataFrame]:
    X = jan_ref[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = jan_ref[TARGET]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    reg = ensemble.RandomForestRegressor(
        random_state=random_state, n_estimators=n_estimators
    )
    reg.fit(X_train, y_train)
    # Attach target & predictions for Evidently
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["target"] = y_train
    X_train["prediction"] = reg.predict(
        X_train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    )
    X_test["target"] = y_test
    X_test["prediction"] = reg.predict(
        X_test[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    )
    return reg, X_train, X_test


def _compute_mae(df_with_pred: pd.DataFrame) -> float:
    return float(np.mean(np.abs(df_with_pred["target"] - df_with_pred["prediction"])))


# -----------------------------
# Main script
# -----------------------------

def main() -> None:
    # Parsing of input arguments
    parser = argparse.ArgumentParser(
        description="Generate all Evidently reports for the Bike Sharing exam (single run)."
    )
    parser.add_argument(
        "--outdir", default="reports", help="Base output directory (default: reports)"
    )
    parser.add_argument(
        "--project",
        default="bike-jan-feb-2011",
        help="Project folder name to group all reports (default: bike-jan-feb-2011)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=50,
        help="RandomForest n_estimators (default: 50)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    args = parser.parse_args()

    # Logging setup INFO level with template (date/level/message)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Report dir initialization
    project_dir = os.path.join(args.outdir, args.project)
    os.makedirs(project_dir, exist_ok=True)

    # Step 1 — Fetch raw data and prepare them for time series
    logging.info("Downloading and preparing dataset...")
    raw = _fetch_data()

    # Splits
    jan_full = raw.loc["2011-01-01 00:00:00":"2011-01-31 23:00:00"]

    # Step 2 — Validation report (train vs test on Jan 1–31)
    logging.info("Training validation model on Jan 1–31...")
    reg, X_train, X_test = _train_validation_model(
        jan_full, args.n_estimators, args.random_state
    )
    cm_val = _build_column_mapping(
        NUMERICAL_FEATURES, CATEGORICAL_FEATURES, "target", "prediction"
    )
    _regression_report(
        X_train,
        X_test,
        cm_val,
        os.path.join(project_dir, "02_validation.html"),
        os.path.join(project_dir, "02_validation.json"),
    )

    # Step 3 — Production model retrained on full January
    logging.info("Training production model on full January..")
    reg.fit(jan_full[NUMERICAL_FEATURES + CATEGORICAL_FEATURES], jan_full[TARGET])
    jan_full_rep = jan_full[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    jan_full_rep["target"] = jan_full[TARGET]
    jan_full_rep["prediction"] = reg.predict(
        jan_full[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    )
    cm_prod = _build_column_mapping(
        NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET, PREDICTION
    )
    _regression_report(
        None,
        jan_full_rep,
        cm_prod,
        os.path.join(project_dir, "03_prod_january.html"),
        os.path.join(project_dir, "03_prod_january.json"),
    )

    # Step 4 — Weekly monitoring reports on February weeks
    logging.info("Building weekly monitoring reports (Feb weeks)…")
    # Use January (full) as reference for week comparisons
    jan_ref_for_weeks = jan_full_rep

    mae_by_week: dict[str, float] = {}
    for wk, (start, end) in WEEKS.items():
        week_df = raw.loc[start:end]
        week_rep = week_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
        week_rep["target"] = week_df[TARGET]
        week_rep["prediction"] = reg.predict(
            week_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        )
        _regression_report(
            jan_ref_for_weeks,
            week_rep,
            cm_prod,
            os.path.join(project_dir, f"04_{wk}_regression.html"),
            os.path.join(project_dir, f"04_{wk}_regression.json"),
        )
        mae_by_week[wk] = _compute_mae(week_rep)

    worst_week = max(mae_by_week, key=mae_by_week.get)  # type: ignore
    worst_range = WEEKS[worst_week]
    logging.info(
        "Worst week by MAE: %s -> %s (%s)",
        worst_week, mae_by_week[worst_week], worst_range
    )

    # Step 5 — Target drift on the worst week (vs January)
    logging.info("Building target drift report on %s…", worst_week)
    worst_df = raw.loc[worst_range[0]:worst_range[1]]
    _target_drift_report(
        jan_full,
        worst_df,
        TARGET,
        os.path.join(project_dir, f"05_target_drift_{worst_week}.html"),
        os.path.join(project_dir, f"05_target_drift_{worst_week}.json"),
    )

    # Step 6 — Data drift (numeric only) on last week (week3)
    logging.info("Building numeric-only data drift on week3…")
    last_week = "week3"
    last_range = WEEKS[last_week]
    week3_df = raw.loc[last_range[0]:last_range[1]]
    _data_drift_report(
        jan_full,
        week3_df,
        NUMERICAL_FEATURES,
        os.path.join(project_dir, "06_data_drift_week3_numeric.html"),
        os.path.join(project_dir, "06_data_drift_week3_numeric.json"),
    )

    # Small JSON summary (to help the README step)
    summary = {"mae_by_week": mae_by_week, "worst_week": worst_week, "week_ranges": WEEKS}
    with open(os.path.join(project_dir, "00_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("All done. Reports written to: %s", project_dir)


if __name__ == "__main__":
    main()
