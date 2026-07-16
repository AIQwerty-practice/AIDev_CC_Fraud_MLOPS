"""Reusable configuration, data, evaluation, and artifact helpers."""

import json
import os
from datetime import datetime, timezone

import h2o
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from utils.data_processing import apply_feature_engineering


DEFAULTS = {
    "run_name": None,
    "feature_version": "baseline",
    "threshold": 0.5,
    "sample_frac": 1.0,
    "max_runtime_secs": 0,
    "max_models": 10,
    "seed": 42,
    "stopping_metric": "AUTO",
    "stopping_rounds": 3,
    "sort_metric": "AUTO",
    "balance_classes": True,
    "test_size": 0.2,
    "use_pre_split_test": True,
    "include_algos": [],
    "exclude_algos": [],
    "nfolds": 0,
    "parameters": {},
    "train_path": os.path.join("data", "processed", "train.csv"),
    "test_path": os.path.join("data", "processed", "test.csv"),
}


def load_experiment_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        supplied = yaml.safe_load(stream) or {}
    config = {**DEFAULTS, **supplied}
    missing = sorted({"experiment_name", "algorithm", "target"} - supplied.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    if not 0 <= float(config["threshold"]) <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not 0 < float(config["sample_frac"]) <= 1:
        raise ValueError("sample_frac must be greater than 0 and at most 1")
    if not 0 < float(config["test_size"]) < 1:
        raise ValueError("test_size must be between 0 and 1")
    if any(int(config[key]) < 0 for key in ("max_models", "max_runtime_secs", "nfolds", "stopping_rounds")):
        raise ValueError("max_models, max_runtime_secs, nfolds, and stopping_rounds cannot be negative")
    if config.get("include_algos") and config.get("exclude_algos"):
        raise ValueError("include_algos and exclude_algos are mutually exclusive")
    config["threshold"] = float(config["threshold"])
    config["sample_frac"] = float(config["sample_frac"])
    config["test_size"] = float(config["test_size"])
    config["max_models"] = int(config["max_models"])
    config["max_runtime_secs"] = int(config["max_runtime_secs"])
    config["seed"] = int(config["seed"])
    config["nfolds"] = int(config["nfolds"])
    config["stopping_rounds"] = int(config["stopping_rounds"])
    config["run_name"] = config.get("run_name") or default_run_name(config)
    return config


def default_run_name(config):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    algorithm = str(config["algorithm"]).lower().replace(" ", "-")
    experiment = str(config["experiment_name"]).lower().replace(" ", "-")
    return f"{experiment}-{algorithm}-{timestamp}"


def flatten_config(config, prefix=""):
    """Flatten nested/list values into MLflow-safe parameter strings."""
    flattened = {}
    for key, value in config.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten_config(value, name))
        elif isinstance(value, (list, tuple)):
            flattened[name] = json.dumps(value)
        elif value is None:
            flattened[name] = ""
        else:
            flattened[name] = value
    return flattened


def load_experiment_data(config):
    """Load a predefined test set or create a reproducible stratified split."""
    train_path = config["train_path"]
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    training_data = pd.read_csv(train_path)
    target = config["target"]
    if target not in training_data:
        raise ValueError(f"Target column '{target}' is missing from {train_path}")

    test_path = config.get("test_path")
    if config["use_pre_split_test"]:
        if not test_path or not os.path.exists(test_path):
            raise FileNotFoundError(f"Configured test data not found: {test_path}")
        evaluation_data = pd.read_csv(test_path)
    else:
        training_data, evaluation_data = train_test_split(
            training_data,
            test_size=config["test_size"],
            random_state=config["seed"],
            stratify=training_data[target],
        )

    if target not in evaluation_data:
        raise ValueError(f"Target column '{target}' is missing from evaluation data")
    missing_features = sorted(set(training_data.columns) - set(evaluation_data.columns))
    if missing_features:
        raise ValueError(f"Evaluation data is missing training columns: {', '.join(missing_features)}")

    if config["sample_frac"] < 1:
        training_data, _ = train_test_split(
            training_data,
            train_size=config["sample_frac"],
            random_state=config["seed"],
            stratify=training_data[target],
        )
    training_data = apply_feature_engineering(training_data.copy(), config["feature_version"])
    evaluation_data = apply_feature_engineering(evaluation_data.copy(), config["feature_version"])
    return training_data, evaluation_data


def to_h2o_frames(training_data, evaluation_data, target):
    training_frame = h2o.H2OFrame(training_data)
    evaluation_frame = h2o.H2OFrame(evaluation_data)
    training_frame[target] = training_frame[target].asfactor()
    evaluation_frame[target] = evaluation_frame[target].asfactor()
    predictors = [column for column in training_frame.col_names if column != target]
    return training_frame, evaluation_frame, predictors


def predict_probabilities(model, frame):
    predictions = model.predict(frame).as_data_frame()
    probability_columns = [column for column in predictions if column != "predict"]
    positive_column = "p1" if "p1" in probability_columns else (
        probability_columns[-1] if probability_columns else None
    )
    if positive_column is None:
        raise ValueError("The trained classifier did not produce class probabilities")
    return predictions[positive_column].astype(float).to_numpy(), predictions


def evaluate_binary_classifier(actual, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1_score": f1_score(actual, predicted, zero_division=0),
        "roc_auc": roc_auc_score(actual, probabilities),
    }
    return predicted, metrics


def write_evaluation_artifacts(directory, actual, predicted, probabilities, source_data):
    """Write tabular and visual evaluation artifacts, returning their paths."""
    paths = {}
    confusion = confusion_matrix(actual, predicted, labels=[0, 1])
    confusion_path = os.path.join(directory, "confusion_matrix.csv")
    pd.DataFrame(confusion, index=["actual_0", "actual_1"], columns=["predicted_0", "predicted_1"]).to_csv(confusion_path)
    paths["confusion_matrix"] = confusion_path

    false_positive_rate, true_positive_rate, _ = roc_curve(actual, probabilities)
    roc_path = os.path.join(directory, "roc_curve.png")
    _plot_curve(false_positive_rate, true_positive_rate, "False positive rate", "True positive rate", f"ROC curve (AUC={auc(false_positive_rate, true_positive_rate):.4f})", roc_path)
    paths["roc_curve"] = roc_path

    precision, recall, _ = precision_recall_curve(actual, probabilities)
    pr_path = os.path.join(directory, "precision_recall_curve.png")
    _plot_curve(recall, precision, "Recall", "Precision", f"Precision-Recall curve (AUC={auc(recall, precision):.4f})", pr_path)
    paths["precision_recall_curve"] = pr_path

    sample = source_data.head(100).copy()
    sample["predicted_probability"] = probabilities[: len(sample)]
    sample["predicted_class"] = predicted[: len(sample)]
    sample_path = os.path.join(directory, "prediction_sample.csv")
    sample.to_csv(sample_path, index=False)
    paths["prediction_sample"] = sample_path
    return paths


def _plot_curve(x_values, y_values, x_label, y_label, title, path):
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(x_values, y_values, linewidth=2)
    axis.set(xlabel=x_label, ylabel=y_label, title=title)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()
