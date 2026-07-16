import json
import os

DEFAULT_TARGET_COLUMN = "Class"


def apply_feature_engineering(frame, feature_version="baseline"):
    """Apply deterministic features to either a pandas or H2O frame."""
    if feature_version in (None, "baseline"):
        return frame
    if feature_version != "engineered_v1":
        raise ValueError(f"Unknown feature_version: {feature_version}")
    required = {"V1", "V2", "V3", "V4", "Amount"}
    columns = set(frame.columns if hasattr(frame, "columns") else frame.col_names)
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"engineered_v1 requires columns: {', '.join(missing)}")
    frame["V1_V2_interaction"] = frame["V1"] * frame["V2"]
    frame["V3_V4_interaction"] = frame["V3"] * frame["V4"]
    frame["Amount_abs"] = abs(frame["Amount"])
    return frame


def load_model_metadata():
    artifact_dir = os.getenv("MODEL_ARTIFACT_DIR", "model_artifacts")
    path = os.path.join(artifact_dir, "model_metadata.json")
    if not os.path.exists(path):
        return {"feature_version": "baseline", "threshold": 0.5, "target": DEFAULT_TARGET_COLUMN}
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def separate_id_col(frame):
    """Separate an ID column when present.

    For this credit-card fraud dataset there is normally no ID column, so the
    function usually returns the full frame unchanged.
    """
    candidate_ids = ["id", "ID", "Id", "Customer ID", "Transaction ID"]
    for col in candidate_ids:
        if col in frame.col_names:
            id_frame = frame[[col]]
            x_frame = frame.drop(col)
            return col, id_frame, x_frame
    return None, None, frame


def match_col_types(frame):
    """Match uploaded prediction data to training columns/types.

    - Drops target columns if the user uploads a labeled sample.
    - Adds any missing predictor column as 0.
    - Reorders columns to match the training frame.
    - Applies H2O column types stored during training.
    """
    metadata = load_model_metadata()
    frame = apply_feature_engineering(frame, metadata.get("feature_version", "baseline"))
    artifact_dir = os.getenv("MODEL_ARTIFACT_DIR", "model_artifacts")
    types_path = os.path.join(artifact_dir, "train_col_types.json")
    if not os.path.exists(types_path):
        types_path = "data/processed/train_col_types.json"
    if not os.path.exists(types_path):
        return frame

    with open(types_path, "r", encoding="utf-8") as fp:
        train_types = json.load(fp)

    target_columns = {DEFAULT_TARGET_COLUMN, metadata.get("target", DEFAULT_TARGET_COLUMN)}
    predictor_cols = [c for c in train_types.keys() if c not in target_columns]

    # Drop target if present in uploaded labeled file.
    for target in target_columns:
        if target in frame.col_names:
            frame = frame.drop(target)

    # Add missing predictors as zeros.
    for col in predictor_cols:
        if col not in frame.col_names:
            frame[col] = 0

    # Keep only predictors in training order.
    frame = frame[predictor_cols]

    # Apply H2O types where possible.
    for col in predictor_cols:
        typ = train_types.get(col)
        if typ == "enum":
            frame[col] = frame[col].asfactor()
        elif typ in {"int", "real"}:
            frame[col] = frame[col].asnumeric()

    return frame
