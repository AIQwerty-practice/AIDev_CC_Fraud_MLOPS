"""Single entry point for configuration-driven H2O experiments."""

import argparse
import json
import logging
import os
import shutil
import tempfile

import h2o
import mlflow
import mlflow.h2o
from h2o.exceptions import H2OError
from mlflow.tracking import MlflowClient

from experiment_utils import (
    evaluate_binary_classifier,
    flatten_config,
    load_experiment_config,
    load_experiment_data,
    predict_probabilities,
    to_h2o_frames,
    utc_timestamp,
    write_evaluation_artifacts,
)
from model_factory import leaderboard_as_frame, train_configured_model
from utils.h2o_connection import connect_h2o

LOGGER = logging.getLogger("fraud_experiment")


def environment_value(name, default=None):
    value = os.getenv(name)
    return value if value not in (None, "") else default


def parse_args():
    parser = argparse.ArgumentParser(description="Run a configured H2O/MLflow experiment")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment configuration")
    parser.add_argument("--target", help="Optional target-column override for legacy commands")
    return parser.parse_args()


def log_feature_importance(model, output_directory):
    """Log feature importance when the selected H2O model exposes it."""
    try:
        importance = model.varimp(use_pandas=True)
    except (AttributeError, TypeError, ValueError, H2OError) as exc:
        LOGGER.info("Feature importance is unavailable for %s: %s", model.model_id, exc)
        return None
    if importance is None or importance.empty:
        LOGGER.info("Feature importance is unavailable for %s", model.model_id)
        return None
    path = os.path.join(output_directory, "feature_importance.csv")
    importance.to_csv(path, index=False)
    return path


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)


def publish_deployment_artifacts(model, artifact_directory, metadata, column_types):
    """Publish model pointer and metadata only after registry promotion succeeds."""
    native_directory = os.path.join(artifact_directory, "h2o_model")
    os.makedirs(native_directory, exist_ok=True)
    native_model_path = h2o.save_model(model=model, path=native_directory, force=True)

    staged = {
        "model_path.txt": native_model_path,
        "model_metadata.json": metadata,
        "train_col_types.json": column_types,
    }
    for filename, value in staged.items():
        temporary_path = os.path.join(artifact_directory, f".{filename}.tmp")
        final_path = os.path.join(artifact_directory, filename)
        if isinstance(value, str):
            with open(temporary_path, "w", encoding="utf-8") as stream:
                stream.write(value)
        else:
            write_json(temporary_path, value)
        os.replace(temporary_path, final_path)
    return native_model_path


def run_experiment(config_path, target_override=None):
    config = load_experiment_config(config_path)
    if target_override:
        config["target"] = target_override

    tracking_uri = environment_value("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    artifact_directory = environment_value("MODEL_ARTIFACT_DIR", "model_artifacts")
    os.makedirs(artifact_directory, exist_ok=True)

    LOGGER.info("Loading experiment data for '%s'", config["experiment_name"])
    training_data, evaluation_data = load_experiment_data(config)
    training_frame, evaluation_frame, predictors = to_h2o_frames(
        training_data, evaluation_data, config["target"]
    )
    actual = evaluation_data[config["target"]].astype(int).to_numpy()
    started_at = utc_timestamp()

    mlflow.set_experiment(config["experiment_name"])
    client = MlflowClient()
    with mlflow.start_run(run_name=config["run_name"]) as run:
        mlflow.set_tags({
            "experiment_name": config["experiment_name"],
            "run_name": config["run_name"],
            "started_at_utc": started_at,
            "algorithm": config["algorithm"],
            "feature_version": config["feature_version"],
        })
        mlflow.log_params(flatten_config(config))
        LOGGER.info("Training %s for run %s", config["algorithm"], run.info.run_id)
        model, automl = train_configured_model(
            config, predictors, config["target"], training_frame
        )
        probabilities, _ = predict_probabilities(model, evaluation_frame[predictors])
        predicted, metrics = evaluate_binary_classifier(actual, probabilities, config["threshold"])
        mlflow.log_metrics(metrics)

        metadata = {
            "experiment_name": config["experiment_name"],
            "run_name": config["run_name"],
            "run_id": run.info.run_id,
            "started_at_utc": started_at,
            "completed_at_utc": utc_timestamp(),
            "model_id": model.model_id,
            "algorithm": config["algorithm"],
            "feature_version": config["feature_version"],
            "threshold": config["threshold"],
            "target": config["target"],
            "predictors": predictors,
            "metrics": metrics,
            "config": config,
        }

        with tempfile.TemporaryDirectory() as output_directory:
            evaluation_paths = write_evaluation_artifacts(
                output_directory, actual, predicted, probabilities, evaluation_data
            )
            leaderboard = leaderboard_as_frame(model, automl)
            for metric_name, metric_value in metrics.items():
                leaderboard[f"evaluation_{metric_name}"] = metric_value
            leaderboard_path = os.path.join(output_directory, "leaderboard.csv")
            leaderboard.to_csv(leaderboard_path, index=False)
            metadata_path = os.path.join(output_directory, "model_metadata.json")
            write_json(metadata_path, metadata)
            config_copy_path = os.path.join(output_directory, "experiment_config.yaml")
            shutil.copyfile(config_path, config_copy_path)
            log_feature_importance(model, output_directory)
            mlflow.log_artifacts(output_directory, artifact_path="experiment")
            LOGGER.info("Logged %d evaluation artifacts", len(evaluation_paths))

        mlflow.h2o.log_model(
            model,
            artifact_path="model",
            input_example=evaluation_data[predictors].head(5),
            metadata={
                "experiment_name": config["experiment_name"],
                "run_name": config["run_name"],
                "feature_version": config["feature_version"],
                "threshold": config["threshold"],
            },
        )

    model_name = environment_value("MODEL_NAME", "fraud-detection-automl")
    model_alias = environment_value("MODEL_ALIAS", "champion")
    registered = mlflow.register_model(f"runs:/{run.info.run_id}/model", model_name)
    client.set_registered_model_alias(model_name, model_alias, registered.version)
    metadata["registered_model_name"] = model_name
    metadata["registered_model_version"] = registered.version
    metadata["registered_model_alias"] = model_alias
    with tempfile.TemporaryDirectory() as output_directory:
        metadata_path = os.path.join(output_directory, "model_metadata.json")
        write_json(metadata_path, metadata)
        with mlflow.start_run(run_id=run.info.run_id):
            mlflow.log_artifact(metadata_path, artifact_path="experiment")
    native_path = publish_deployment_artifacts(
        model, artifact_directory, metadata, training_frame.types
    )
    LOGGER.info(
        "Completed %s; registered %s v%s as @%s; native model=%s; metrics=%s",
        config["experiment_name"], model_name, registered.version, model_alias,
        native_path, metrics,
    )
    return metrics


def main():
    logging.basicConfig(
        level=environment_value("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    external_h2o_cluster = connect_h2o()
    try:
        run_experiment(args.config, args.target)
    except Exception:
        LOGGER.exception("Experiment failed")
        raise
    finally:
        if not external_h2o_cluster:
            h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()
