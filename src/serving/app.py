import os
import pickle
import datetime as dt

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Path to the trained model
MODEL_PATH = "models/model.pkl"
LOG_PATH = "data/logs/predictions.csv"

# --- Pydantic request model ---

class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --- Load model at startup ---

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Run `python src/training/train.py` first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Feature names must match what the model was trained on
FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

# --- FastAPI app ---

app = FastAPI(
    title="Project 3 - Iris Model API",
    description="FastAPI service for Iris predictions with logging for monitoring/drift detection.",
    version="0.1.0",
)

# --- Helper: log prediction to CSV ---

def log_prediction(payload: IrisRequest, prediction: int):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    # Build a single-row DataFrame
    row = {
        "timestamp": dt.datetime.utcnow().isoformat(),
        "sepal_length": payload.sepal_length,
        "sepal_width": payload.sepal_width,
        "petal_length": payload.petal_length,
        "petal_width": payload.petal_width,
        "prediction": int(prediction),
    }

    df = pd.DataFrame([row])

    # If file doesn't exist, write with header; else append without header
    if not os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, index=False)
    else:
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)


# --- Routes ---

@app.get("/")
def read_root():
    return {
        "message": "Iris model API is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: IrisRequest):
    # Convert request to DataFrame with correct feature names
    data = {
        "sepal length (cm)": [request.sepal_length],
        "sepal width (cm)": [request.sepal_width],
        "petal length (cm)": [request.petal_length],
        "petal width (cm)": [request.petal_width],
    }
    df = pd.DataFrame(data, columns=FEATURE_COLUMNS)

    pred = model.predict(df)[0]
    pred_int = int(pred)

    # Log the prediction
    log_prediction(request, pred_int)

    return {
        "prediction": pred_int,
        "class_index": pred_int,
        "detail": "Iris class index according to sklearn's Iris dataset",
    }
