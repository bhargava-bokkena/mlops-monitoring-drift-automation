import json
import os
import subprocess

DRIFT_STATUS_PATH = "reports/drift_status.json"


def should_retrain() -> bool:
    """Return True if drift_status.json says data_drift = true."""
    if not os.path.exists(DRIFT_STATUS_PATH):
        print("[retrain] Drift status file not found. Assuming no drift.")
        return False

    with open(DRIFT_STATUS_PATH, "r") as f:
        status = json.load(f)

    drift_flag = bool(status.get("data_drift", False))
    print(f"[retrain] data_drift from drift_status.json = {drift_flag}")
    return drift_flag


def run_retraining():
    """Call the training script to retrain and overwrite the model."""
    print("[retrain] Drift detected — running retraining pipeline...")

    result = subprocess.run(
        ["python", "src/training/train.py"],
        capture_output=True,
        text=True,
    )

    print("[retrain] --- train.py stdout ---")
    print(result.stdout)

    if result.returncode != 0:
        print("[retrain] --- train.py stderr ---")
        print(result.stderr)
        raise RuntimeError("[retrain] Retraining failed")

    print("[retrain] Retraining complete. New model saved to models/model.pkl")


if __name__ == "__main__":
    print("[retrain] Checking drift status...")
    if should_retrain():
        run_retraining()
    else:
        print("[retrain] No drift detected — skipping retraining.")
