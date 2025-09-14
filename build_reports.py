#!/usr/bin/env python3
"""Build Evidently HTML reports along with workspace storage in one single run.

Outputs (all in the same project):
  - 01_validation.html                      (model validation report (train vs test) on Jan)
  - 02_model_drift_january.html             (model drift monitoring on January full)
  - 03_model_drift_feb_week{1,2,3}.html     (model drift monitoring on the 3 first weeks of Feb)
  - 04a_feb_weeks_audit.json                (Feb weeks MAE audit to identify worst week)
  - 04b_target_drift_feb_week{1,2,3}.html   (target drift monitoring on Feb worst week)
  - 05_data_drift_feb_week3_numeric.html    (data drift monitoring on last week, numeric only)
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
from sklearn import ensemble

from evidently.metric_preset import (
    RegressionPreset,
    DataDriftPreset,
    TargetDriftPreset,
)
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.ui.workspace import Workspace

# Silence noisy warnings as in the statement examples
warnings.filterwarnings("ignore")

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
)

TARGET = "cnt"
PREDICTION = "cnt_predicted"
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

# Weeks as required (asymmetric; may overlap)
WEEKS = {
    "week1": ("2011-01-29 00:00:00", "2011-02-07 23:00:00"),
    "week2": ("2011-02-07 00:00:00", "2011-02-14 23:00:00"),
    "week3": ("2011-02-15 00:00:00", "2011-02-21 23:00:00"),
}


# -----------------------------
# Internal helper functions
# -----------------------------

# ---- Data management ----
def fetch_data() -> pd.DataFrame:
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


# ---- Report generators ----
def save_html(rep: Report, out_html: str) -> None:
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    rep.save_html(out_html)


def generate_regression_performance_report(
    reference_data: pd.DataFrame | None,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping,
) -> Report:
    """Generates a regression performance report using Evidently."""
    rep = Report(metrics=[RegressionPreset()])
    rep.run(
        reference_data=reference_data.sort_index() if reference_data is not None else None,
        current_data=current_data.sort_index(),
        column_mapping=column_mapping,
    )
    return rep


def generate_target_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target: str,
) -> Report:
    """Generates a target drift report using Evidently."""
    cm = ColumnMapping()
    cm.target = target
    rep = Report(metrics=[TargetDriftPreset()])
    rep.run(
        reference_data=reference_data[[target]].sort_index(),
        current_data=current_data[[target]].sort_index(),
        column_mapping=cm,
    )
    return rep


def generate_data_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    numerical_features: list[str],
) -> Report:
    """Generates a data drift (numeric-only) report using Evidently."""
    cm = ColumnMapping()
    cm.numerical_features = numerical_features
    rep = Report(metrics=[DataDriftPreset()])
    rep.run(
        reference_data=reference_data[numerical_features].sort_index(),
        current_data=current_data[numerical_features].sort_index(),
        column_mapping=cm,
    )
    return rep


def add_report_to_workspace(
    workspace: Workspace,
    project_name: str,
    project_description: str,
    report: Report,
) -> None:
    """Adds a report to an existing or new project in a local workspace."""
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description
        project.save()

    workspace.add_report(project.id, report)
    logging.info("Report added to Evidently workspace project=%s", project_name)


# ---- Training & metrics ----
def train_validation_model(
    jan_ref: pd.DataFrame,
    n_estimators: int,
    random_state: int,
    test_ratio: float = 0.30,
) -> tuple[ensemble.RandomForestRegressor, pd.DataFrame, pd.DataFrame]:
    """
    Time-aware split: first (1 - test_ratio) chunk is train, last chunk is test.
    Avoids leakage for time series.
    """
    # Ensure chronological order
    df = jan_ref.sort_index()
    n = len(df)
    cut = max(1, int(n * (1.0 - test_ratio)))
    logging.info(f"len(df) = [{n}] / cut_index = [{cut}]")
    train_df = df.iloc[:cut]
    test_df = df.iloc[cut:]

    # Train on past, evaluate on future
    X_train = train_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET]

    reg = ensemble.RandomForestRegressor(
        random_state=random_state, n_estimators=n_estimators
    )
    reg.fit(X_train, y_train)

    # Attach target & predictions for Evidently
    train_rep = X_train.copy()
    test_rep = X_test.copy()
    train_rep[TARGET] = y_train
    train_rep[PREDICTION] = reg.predict(X_train)
    test_rep[TARGET] = y_test
    test_rep[PREDICTION] = reg.predict(X_test)

    return reg, train_rep, test_rep


def _compute_mae(df_with_pred: pd.DataFrame) -> float:
    return float(np.mean(np.abs(df_with_pred[TARGET] - df_with_pred[PREDICTION])))


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
    # Local Evidently UI
    parser.add_argument(
        "--ui-local",
        action="store_true",
        help="Also log reports into a local Evidently workspace/project",
    )
    parser.add_argument(
        "--ui-workspace",
        default=".evidently_workspace",
        help="Local Evidently workspace path (default: .evidently_workspace)",
    )
    parser.add_argument(
        "--ui-project-name",
        default=None,
        help="Evidently project name (default: --project)",
    )
    parser.add_argument(
        "--ui-project-desc",
        default="Bike Sharing exam — Jan-Feb 2011 (local)",
        help="Evidently project description",
    )

    args = parser.parse_args()

    # Logging setup INFO level with template (date/level/message)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Report dir initialization
    project_dir = os.path.join(args.outdir, args.project)
    os.makedirs(project_dir, exist_ok=True)

    # Optional — init local UI
    ui_ws: Workspace | None = None
    ui_proj_name = args.ui_project_name or args.project
    ui_proj_desc = args.ui_project_desc
    if args.ui_local:
        ui_ws = Workspace.create(args.ui_workspace)

    # Step 1 — Fetch raw data and prepare them for time series
    logging.info("Downloading and preparing dataset...")
    raw = fetch_data()

    # Splits — January reference is 01 -> 28 inclusive (as required)
    jan_full = raw.loc["2011-01-01 00:00:00":"2011-01-28 23:00:00"]

    # Step 2 — Validation report (train vs test on Jan 1–28)
    logging.info("Training validation model on Jan 1–28...")
    reg, X_train, X_test = train_validation_model(
        jan_full, args.n_estimators, args.random_state
    )
    cm_val = _build_column_mapping(
        NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET, PREDICTION
    )
    rep = generate_regression_performance_report(
        reference_data=X_train,
        current_data=X_test,
        column_mapping=cm_val,
    )
    save_html(rep, os.path.join(project_dir, "01_validation.html"))
    if ui_ws:
        add_report_to_workspace(ui_ws, ui_proj_name, ui_proj_desc, rep)

    # Step 3 — Production model retrained on full January (1–28)
    logging.info("Training production model on full January...")
    reg.fit(jan_full[NUMERICAL_FEATURES + CATEGORICAL_FEATURES], jan_full[TARGET])
    jan_full_rep = jan_full[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    jan_full_rep[TARGET] = jan_full[TARGET]
    jan_full_rep[PREDICTION] = reg.predict(
        jan_full[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    )
    cm_prod = _build_column_mapping(
        NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET, PREDICTION
    )
    rep = generate_regression_performance_report(
        reference_data=None,
        current_data=jan_full_rep,
        column_mapping=cm_prod,
    )
    save_html(rep, os.path.join(project_dir, "02_model_drift_january.html"))
    if ui_ws:
        add_report_to_workspace(ui_ws, ui_proj_name, ui_proj_desc, rep)

    # Step 4 — Weekly monitoring reports on February weeks
    logging.info("Building weekly monitoring reports (Feb weeks)...")
    # Use January (1–28) as reference for week comparisons
    jan_ref_for_weeks = jan_full_rep

    mae_by_week: dict[str, float] = {}
    for wk, (start, end) in WEEKS.items():
        week_df = raw.loc[start:end]
        week_rep = week_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
        week_rep[TARGET] = week_df[TARGET]
        week_rep[PREDICTION] = reg.predict(
            week_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        )
        rep = generate_regression_performance_report(
            reference_data=jan_ref_for_weeks,
            current_data=week_rep,
            column_mapping=cm_prod,
        )
        save_html(rep, os.path.join(project_dir, f"03_model_drift_feb_{wk}.html"))
        if ui_ws:
            add_report_to_workspace(ui_ws, ui_proj_name, ui_proj_desc, rep)
        mae_by_week[wk] = _compute_mae(week_rep)

    # Step 5 — Target drift on the worst week (vs January)
    worst_week = max(mae_by_week, key=mae_by_week.get)  # type: ignore
    worst_range = WEEKS[worst_week]
    logging.info(
        "Worst week by MAE: %s -> %s (%s)",
        worst_week, mae_by_week[worst_week], worst_range
    )

    # Save concise audit for README
    summary = {"mae_by_week": mae_by_week, "worst_week": worst_week, "week_ranges": WEEKS}
    with open(os.path.join(project_dir, "04a_feb_weeks_audit.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logging.info("Building target drift report on %s…", worst_week)
    worst_df = raw.loc[worst_range[0]:worst_range[1]]
    rep = generate_target_drift_report(
        reference_data=jan_full,
        current_data=worst_df,
        target=TARGET,
    )
    save_html(rep, os.path.join(project_dir, f"04b_target_drift_feb_{worst_week}.html"))
    if ui_ws:
        add_report_to_workspace(ui_ws, ui_proj_name, ui_proj_desc, rep)

    # Step 6 — Data drift (numeric only) on last week (week3)
    logging.info("Building numeric-only data drift on week3...")
    last_week = "week3"
    last_range = WEEKS[last_week]
    week3_df = raw.loc[last_range[0]:last_range[1]]
    rep = generate_data_drift_report(
        reference_data=jan_full,
        current_data=week3_df,
        numerical_features=NUMERICAL_FEATURES,
    )
    save_html(rep, os.path.join(project_dir, "05_data_drift_feb_week3_numeric.html"))
    if ui_ws:
        add_report_to_workspace(ui_ws, ui_proj_name, ui_proj_desc, rep)

    logging.info("All done. Reports written to: %s", project_dir)


if __name__ == "__main__":
    main()
