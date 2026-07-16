# ===========================
# Module: Backend setup (H2O + MLflow + FastAPI)
# Loads the @champion model from the MLflow Model Registry and serves predictions.
# Original author: Kenneth Leung (modernised for MLflow 2.x + Model Registry)
# ===========================
# Run locally: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
import io
import logging
import os

import h2o
import mlflow
import mlflow.h2o
import pandas as pd
from fastapi import FastAPI, File, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

from utils.data_processing import load_model_metadata, match_col_types, separate_id_col
from utils.h2o_connection import connect_h2o

MODEL_NAME = os.getenv("MODEL_NAME", "fraud-detection-automl")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_ARTIFACT_DIR = os.getenv("MODEL_ARTIFACT_DIR", "model_artifacts")
TARGET_COL = "Class"
LOGGER = logging.getLogger("fraud_api")

app = FastAPI(title="End-to-End AutoML - Credit Card Fraud Detection")

allowed_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if origin.strip()
]
if "*" in allowed_origins:
    raise ValueError("ALLOWED_ORIGINS must list explicit trusted origins; '*' is not permitted")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)

# Initiate H2O instance and point MLflow at the tracking server
connect_h2o()
if TRACKING_URI:
    mlflow.set_tracking_uri(TRACKING_URI)

# Load the best model from the MLflow Model Registry (alias-based reference)
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
LOGGER.info("Loading model from registry: %s", model_uri)
try:
    best_model = mlflow.h2o.load_model(model_uri)
except Exception as exc:
    model_path_file = os.path.join(MODEL_ARTIFACT_DIR, "model_path.txt")
    if not os.path.exists(model_path_file):
        raise exc
    with open(model_path_file, "r", encoding="utf-8") as fp:
        native_model_path = fp.read().strip()
    LOGGER.warning("Registry load failed; loading native H2O model from %s: %s", native_model_path, exc)
    best_model = h2o.load_model(native_model_path)
LOGGER.info("Model loaded successfully")
MODEL_METADATA = load_model_metadata()
PREDICTION_THRESHOLD = float(MODEL_METADATA.get("threshold", 0.5))
TARGET_COL = MODEL_METADATA.get("target", TARGET_COL)
LOGGER.info(
    "Using prediction threshold %s and feature version %s",
    PREDICTION_THRESHOLD,
    MODEL_METADATA.get("feature_version", "baseline"),
)


@app.post("/predict")
async def predict(file: bytes = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty")
    try:
        test_df = pd.read_csv(io.BytesIO(file))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {exc}") from exc
    if test_df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV contains no rows")
    LOGGER.info("Predicting %d uploaded rows", len(test_df))
    test_h2o = h2o.H2OFrame(test_df)

    # Separate ID column (if any)
    id_name, X_id, X_h2o = separate_id_col(test_h2o)

    # Match test set column types with train set
    X_h2o = match_col_types(X_h2o)

    # Generate predictions with best model (output is H2O frame)
    preds = best_model.predict(X_h2o).as_data_frame()
    probability_cols = [col for col in preds.columns if col != "predict"]
    positive_probability_col = "p1" if "p1" in probability_cols else (probability_cols[-1] if probability_cols else None)

    ids = X_id.as_data_frame()[id_name].tolist() if id_name is not None else list(range(1, len(preds) + 1))
    results = []
    for row_id, (_, row) in zip(ids, preds.iterrows()):
        result = {
            "id": row_id,
            "prediction": int(float(row[positive_probability_col]) >= PREDICTION_THRESHOLD) if positive_probability_col else int(row["predict"]),
            "probability": float(row[positive_probability_col]) if positive_probability_col else None,
        }
        if probability_cols:
            result["probabilities"] = {col: float(row[col]) for col in probability_cols}
        results.append(result)

    response = {
        "target_column": TARGET_COL,
        "positive_class": 1,
        "positive_label": "Fraud",
        "threshold": PREDICTION_THRESHOLD,
        "predictions": results,
    }
    if id_name is not None:
        response["id_column"] = id_name

    json_compatible_item_data = jsonable_encoder(response)
    return JSONResponse(content=json_compatible_item_data)


@app.get("/health")
async def health():
    return PlainTextResponse("OK")


@app.get("/")
async def main():
    content = """
    <body>
    <h2>Welcome to the End-to-End AutoML Pipeline for Credit Card Fraud Detection</h2>
    <p>The H2O model and FastAPI instance have been set up successfully.</p>
    <p>Interactive API docs: <a href="/docs">/docs</a></p>
    <p>Open the configured Streamlit interface to submit credit-card transaction prediction requests.</p>
    </body>
    """
    return HTMLResponse(content=content)
