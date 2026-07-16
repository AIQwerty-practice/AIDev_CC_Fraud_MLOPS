"""Factories and training adapters for H2O AutoML and explicit estimators."""

from h2o.automl import H2OAutoML, get_leaderboard
from h2o.estimators import (
    H2OGradientBoostingEstimator,
    H2ORandomForestEstimator,
    H2OXGBoostEstimator,
)


def normalize_algorithm(name):
    return str(name).replace("_", "").replace("-", "").replace(" ", "").lower()


def create_model(algorithm, parameters=None):
    """Create a supported explicit H2O estimator."""
    factories = {
        "gbm": H2OGradientBoostingEstimator,
        "randomforest": H2ORandomForestEstimator,
        "drf": H2ORandomForestEstimator,
        "xgboost": H2OXGBoostEstimator,
    }
    normalized = normalize_algorithm(algorithm)
    if normalized not in factories:
        supported = "GBM, RandomForest (DRF), XGBoost, AutoML"
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Supported: {supported}")
    return factories[normalized](**dict(parameters or {}))


def create_automl(config):
    """Create H2O AutoML while omitting mutually exclusive empty algorithm lists."""
    options = {
        "max_runtime_secs": config["max_runtime_secs"],
        "max_models": config["max_models"],
        "seed": config["seed"],
        "stopping_metric": config["stopping_metric"],
        "stopping_rounds": config["stopping_rounds"],
        "balance_classes": config["balance_classes"],
        "nfolds": config["nfolds"],
        "sort_metric": config["sort_metric"],
        "verbosity": config.get("verbosity", "info"),
    }
    include_algos = config.get("include_algos") or []
    exclude_algos = config.get("exclude_algos") or []
    if include_algos:
        options["include_algos"] = include_algos
    elif exclude_algos:
        options["exclude_algos"] = exclude_algos
    return H2OAutoML(**options)


def train_configured_model(config, predictors, target, training_frame):
    """Train AutoML or one explicit estimator and return its deployable model."""
    if normalize_algorithm(config["algorithm"]) == "automl":
        trainer = create_automl(config)
        trainer.train(x=predictors, y=target, training_frame=training_frame)
        return trainer.leader, trainer

    model_parameters = dict(config.get("parameters") or {})
    model_parameters.setdefault("seed", config["seed"])
    model_parameters.setdefault("balance_classes", config["balance_classes"])
    model_parameters.setdefault("stopping_metric", config["stopping_metric"])
    model_parameters.setdefault("stopping_rounds", config["stopping_rounds"])
    if config["max_runtime_secs"] > 0:
        model_parameters.setdefault("max_runtime_secs", config["max_runtime_secs"])
    if config["nfolds"] > 0:
        model_parameters.setdefault("nfolds", config["nfolds"])
    model = create_model(config["algorithm"], model_parameters)
    model.train(x=predictors, y=target, training_frame=training_frame)
    return model, None


def leaderboard_as_frame(model, automl=None):
    """Return a comparable leaderboard for both AutoML and estimator runs."""
    if automl is not None:
        return get_leaderboard(automl, extra_columns="ALL").as_data_frame()
    return model._model_json["output"]["model_summary"].as_data_frame()
