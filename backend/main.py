# ===========================
# Module: Backend setup (H2O + MLflow + FastAPI)
# Loads the @champion model from the MLflow Model Registry and serves predictions.
# Original author: Kenneth Leung (modernised for MLflow 2.x + Model Registry)
# ===========================
# Run locally: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
import io
import os

import h2o
import mlflow
import mlflow.h2o
import pandas as pd
from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from utils.data_processing import match_col_types, separate_id_col

MODEL_NAME = os.getenv("MODEL_NAME", "fraud-detection-automl")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_ARTIFACT_DIR = os.getenv("MODEL_ARTIFACT_DIR", "model_artifacts")
TARGET_COL = "Class"

app = FastAPI(title="End-to-End AutoML - Credit Card Fraud Detection")

# Initiate H2O instance and point MLflow at the tracking server
h2o.init()
if TRACKING_URI:
    mlflow.set_tracking_uri(TRACKING_URI)

# Load the best model from the MLflow Model Registry (alias-based reference)
model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
print(f"Loading model from registry: {model_uri}")
try:
    best_model = mlflow.h2o.load_model(model_uri)
except Exception as exc:
    model_path_file = os.path.join(MODEL_ARTIFACT_DIR, "model_path.txt")
    if not os.path.exists(model_path_file):
        raise exc
    with open(model_path_file, "r", encoding="utf-8") as fp:
        native_model_path = fp.read().strip()
    print(f"Registry load failed; loading native H2O model from {native_model_path}")
    best_model = h2o.load_model(native_model_path)
print("Model loaded successfully")


@app.post("/predict")
async def predict(file: bytes = File(...)):
    print('[+] Initiate Prediction')
    file_obj = io.BytesIO(file)
    test_df = pd.read_csv(file_obj)
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
            "prediction": int(row["predict"]),
            "probability": float(row[positive_probability_col]) if positive_probability_col else None,
        }
        if probability_cols:
            result["probabilities"] = {col: float(row[col]) for col in probability_cols}
        results.append(result)

    response = {
        "target_column": TARGET_COL,
        "positive_class": 1,
        "positive_label": "Fraud",
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
    <p>Open the Streamlit UI (http://localhost:8501) to submit credit-card transaction prediction requests.</p>
    </body>
    """
    return HTMLResponse(content=content)
