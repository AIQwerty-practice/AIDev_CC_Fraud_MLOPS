import json
import os

TARGET_COLUMNS = {"Class"}


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
    artifact_dir = os.getenv("MODEL_ARTIFACT_DIR", "model_artifacts")
    types_path = os.path.join(artifact_dir, "train_col_types.json")
    if not os.path.exists(types_path):
        types_path = "data/processed/train_col_types.json"
    if not os.path.exists(types_path):
        return frame

    with open(types_path, "r", encoding="utf-8") as fp:
        train_types = json.load(fp)

    predictor_cols = [c for c in train_types.keys() if c not in TARGET_COLUMNS]

    # Drop target if present in uploaded labeled file.
    for target in TARGET_COLUMNS:
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
