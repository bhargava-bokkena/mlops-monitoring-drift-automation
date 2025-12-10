import os
import sys
import json

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

REFERENCE_PATH = "data/processed/reference.csv"
LOG_PATH = "data/logs/predictions.csv"
REPORT_DIR = "reports"
REPORT_PATH = os.path.join(REPORT_DIR, "drift_report.html")
DRIFT_STATUS_PATH = os.path.join(REPORT_DIR, "drift_status.json")


def load_reference_data() -> pd.DataFrame:
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(
            f"Reference data not found at {REFERENCE_PATH}. "
            f"Run `python src/training/train.py` first."
        )
    print(f"[monitor] Loading reference data from {REFERENCE_PATH}")
    return pd.read_csv(REFERENCE_PATH)


def load_current_data() -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(
            f"Log file not found at {LOG_PATH}. "
            f"Hit the /predict endpoint a few times to generate logs."
        )
    print(f"[monitor] Loading current data from {LOG_PATH}")
    curr = pd.read_csv(LOG_PATH)

    if curr.shape[0] < 10:
        print(
            f"[monitor] Warning: Only {curr.shape[0]} rows in {LOG_PATH}. "
            "You may want to generate more predictions for a more meaningful report."
        )

    return curr


def prepare_datasets():
    reference = load_reference_data()
    current_logs = load_current_data()

    feature_cols_ref = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    # --- Reference: features + target ---
    ref_df = reference[feature_cols_ref + ["target"]].copy()

    # --- Current logs: features + prediction ---
    curr_df = current_logs[
        ["sepal_length", "sepal_width", "petal_length", "petal_width", "prediction"]
    ].copy()

    # Rename current feature columns to match reference names
    rename_map = {
        "sepal_length": "sepal length (cm)",
        "sepal_width": "sepal width (cm)",
        "petal_length": "petal length (cm)",
        "petal_width": "petal width (cm)",
    }
    curr_df.rename(columns=rename_map, inplace=True)

    # 🔑 Ensure BOTH datasets have BOTH `target` and `prediction`
    # For reference, use target as stand-in prediction
    ref_df["prediction"] = ref_df["target"]

    # For current, use prediction as stand-in target
    curr_df["target"] = curr_df["prediction"]

    print(f"[monitor] Reference shape: {ref_df.shape}")
    print(f"[monitor] Current   shape: {curr_df.shape}")

    return ref_df, curr_df


def build_column_mapping():
    feature_cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    column_mapping = ColumnMapping()
    column_mapping.numerical_features = feature_cols
    column_mapping.categorical_features = []
    column_mapping.target = "target"
    column_mapping.prediction = "prediction"

    print("[monitor] Column mapping configured.")
    return column_mapping


def run_drift_report():
    print("[monitor] Preparing datasets...")
    ref_df, curr_df = prepare_datasets()
    column_mapping = build_column_mapping()

    report = Report(
        metrics=[DataDriftPreset(), TargetDriftPreset()]
    )

    print("[monitor] Running Evidently drift report...")
    report.run(
        reference_data=ref_df,
        current_data=curr_df,
        column_mapping=column_mapping,
    )

    # --- Save HTML report ---
    os.makedirs(REPORT_DIR, exist_ok=True)
    report.save_html(REPORT_PATH)
    print(f"[monitor] Drift report saved to {REPORT_PATH}")

    # --- Extract drift results safely ---
    results = report.as_dict()

    drift_flag = False  # default

    try:
        metrics = results.get("metrics", [])
        for m in metrics:
            metric_name = m.get("metric") or m.get("metric_name")
            if metric_name and "DataDrift" in metric_name:
                result = m.get("result", {})
                drift_flag = bool(result.get("dataset_drift", False))
                break
    except Exception as e:
        print(f"[monitor] Warning: could not parse drift flag from report.as_dict(): {e}")
        # keep drift_flag = False

    drift_json = {"data_drift": drift_flag}

    try:
        with open(DRIFT_STATUS_PATH, "w") as f:
            json.dump(drift_json, f, indent=2)
        print(f"[monitor] Drift status saved to {DRIFT_STATUS_PATH}")
        print(f"[monitor] Drift detected: {drift_flag}")
    except Exception as e:
        print(f"[monitor] Error while writing drift_status.json: {e}")

    print("[monitor] Done. Open the HTML file in a browser to view the drift dashboard.")


if __name__ == "__main__":
    print("[monitor] Starting monitoring job...")
    try:
        run_drift_report()
    except Exception as e:
        print(f"[monitor] Error while running monitoring: {e}")
        sys.exit(1)
