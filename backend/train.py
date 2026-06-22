# =========================================
# H2O AutoML Training with MLflow Tracking
# - Logs run + metrics to the MLflow Tracking Server
# - Registers the best model in the MLflow Model Registry under an alias
# Original author: Kenneth Leung (modernised for MLflow 2.x + Model Registry)
# =========================================
import argparse
import json
import os
import tempfile

import h2o
from h2o.automl import H2OAutoML, get_leaderboard

import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient


def env(name, default):
    """Read an env var, falling back to a default (empty string treated as unset)."""
    value = os.getenv(name)
    return value if value not in (None, "") else default


def parse_args():
    parser = argparse.ArgumentParser(description="H2O AutoML Training and MLflow Tracking")
    parser.add_argument('--name', '--experiment_name', metavar='',
                        default=env('EXPERIMENT_NAME', 'credit-card-fraud-automl'),
                        help='Name of Experiment. Default is credit-card-fraud-automl', type=str)
    parser.add_argument('--target', '--t', metavar='', required=True,
                        help='Name of Target Column (y)', type=str)
    parser.add_argument('--models', '--m', metavar='',
                        default=int(env('AUTOML_MAX_MODELS', '10')),
                        help='Number of AutoML models to train. Default is 10', type=int)
    parser.add_argument('--runtime', metavar='',
                        default=int(env('AUTOML_MAX_RUNTIME_SECS', '0')),
                        help='Max AutoML runtime in seconds (0 = no limit). Default is 0', type=int)
    parser.add_argument('--sample-frac', metavar='',
                        default=float(env('AUTOML_SAMPLE_FRAC', '1.0')),
                        help='Fraction of training rows to use (speeds up smoke tests). Default is 1.0', type=float)
    return parser.parse_args()


def main():
    args = parse_args()

    # Point MLflow at the tracking server (defaults to local ./mlruns if unset)
    tracking_uri = env('MLFLOW_TRACKING_URI', None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_name = env('MODEL_NAME', 'fraud-detection-automl')
    model_alias = env('MODEL_ALIAS', 'champion')
    artifact_dir = env('MODEL_ARTIFACT_DIR', 'model_artifacts')
    os.makedirs(artifact_dir, exist_ok=True)

    # Initiate H2O cluster
    h2o.init()

    client = MlflowClient()

    # Create (or reuse) the MLflow experiment
    experiment = client.get_experiment_by_name(args.name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(args.name)
        experiment = client.get_experiment(experiment_id)
    mlflow.set_experiment(args.name)

    print(f"Name: {args.name}")
    print(f"Experiment_id: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print(f"Tracking uri: {mlflow.get_tracking_uri()}")

    train_path = os.path.join('data', 'processed', 'train.csv')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f'Training data not found: {train_path}')

    # Import data directly as H2O frame (default location is data/processed)
    main_frame = h2o.import_file(path=train_path)

    # Optional row sampling to keep Docker Desktop smoke tests fast
    if 0 < args.sample_frac < 1.0:
        main_frame = main_frame.split_frame(ratios=[args.sample_frac], seed=42)[0]
        print(f'Sampled training frame to {args.sample_frac:.0%} -> {main_frame.nrow} rows')

    # Save column data types of H2O frame (for matching with test set during prediction)
    col_types_path = os.path.join(artifact_dir, 'train_col_types.json')
    with open(col_types_path, 'w', encoding='utf-8') as fp:
        json.dump(main_frame.types, fp)

    # Set predictor and target columns
    target = args.target
    predictors = [n for n in main_frame.col_names if n != target]

    # Factorize target variable so that AutoML tackles a classification problem
    main_frame[target] = main_frame[target].asfactor()

    with mlflow.start_run() as run:
        aml = H2OAutoML(
            max_models=args.models,
            max_runtime_secs=args.runtime,
            seed=42,
            balance_classes=True,        # Target classes imbalanced
            sort_metric='logloss',
            verbosity='info',
            exclude_algos=['GLM', 'DRF'],
        )

        aml.train(x=predictors, y=target, training_frame=main_frame)

        # Log parameters and metrics
        mlflow.log_param("max_models", args.models)
        mlflow.log_param("max_runtime_secs", args.runtime)
        mlflow.log_param("sample_frac", args.sample_frac)
        mlflow.log_metric("log_loss", aml.leader.logloss())
        mlflow.log_metric("AUC", aml.leader.auc())

        # Log the best model (mlflow.h2o provides the API for logging & loading H2O models)
        mlflow.h2o.log_model(aml.leader, artifact_path="model")
        model_uri = mlflow.get_artifact_uri("model")
        print(f'AutoML best model saved in {model_uri}')

        native_model_dir = os.path.join(artifact_dir, 'h2o_model')
        os.makedirs(native_model_dir, exist_ok=True)
        native_model_path = h2o.save_model(
            model=aml.leader,
            path=native_model_dir,
            force=True,
        )
        with open(os.path.join(artifact_dir, 'model_path.txt'), 'w', encoding='utf-8') as fp:
            fp.write(native_model_path)
        mlflow.log_artifact(col_types_path, artifact_path="model")
        print(f'Native H2O model exported for inference at {native_model_path}')

        # Log the leaderboard as an MLflow artifact (no more brittle local mlruns/ path)
        lb = get_leaderboard(aml, extra_columns='ALL').as_data_frame()
        with tempfile.TemporaryDirectory() as tmp:
            lb_path = os.path.join(tmp, 'leaderboard.csv')
            lb.to_csv(lb_path, index=False)
            mlflow.log_artifact(lb_path, artifact_path="model")
        print('Leaderboard logged as MLflow artifact')

    # Register the model in the Model Registry and tag it with an alias
    registered = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=model_name)
    client.set_registered_model_alias(name=model_name, alias=model_alias, version=registered.version)
    print(f'Registered model "{model_name}" v{registered.version} with alias @{model_alias}')


if __name__ == "__main__":
    main()
