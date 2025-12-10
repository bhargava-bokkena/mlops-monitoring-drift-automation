# MLOps Project 3 – Monitoring, Drift Detection & Automated Retraining

This project implements production-style **model monitoring**, **data drift detection**, and **automated retraining** around a deployed ML model.

This repository demonstrates:
- Logging prediction traffic from an inference API
- Detecting data/prediction drift with batch monitoring jobs
- Automatically retraining and updating a model when drift persists

## Architecture

```mermaid
flowchart LR
    subgraph Online_Inference
        A[Client / App] --> B[FastAPI Model Service]
        B --> C[(Prediction Log Store)]
    end

    subgraph Monitoring
        C --> D[Batch Monitoring Job (Evidently)]
        D --> E[Reports + Alerts (logs/HTML)]
    end

    subgraph Training_and_Retraining
        C --> F[Retraining Pipeline]
        F --> G[Model Registry (MLflow) / Model Artifacts]
        G --> B
    end
