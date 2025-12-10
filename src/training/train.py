import os
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = "models/model.pkl"
REFERENCE_PATH = "data/processed/reference.csv"
MLFLOW_TRACKING_URI = "file:./mlruns"


def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame  # features + target
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42), df


def train_model():
    (X_train, X_test, y_train, y_test), full_df = load_data()

    # --- Save reference data for monitoring ---
    os.makedirs("data/processed", exist_ok=True)
    full_df.to_csv(REFERENCE_PATH, index=False)
    print(f"Reference data saved to {REFERENCE_PATH}")

    # --- Train model ---
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc:.4f}")

    # --- Save model artifact ---
    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")

    # --- Track with MLflow (local ./mlruns) ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("project3_baseline_training")

    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    print("MLflow run logged.")


if __name__ == "__main__":
    train_model()
