# Project 3 - Credit Card Fraud Detection with H2O AutoML + MLflow

This is a complete drop-in project folder for the teacher H2O/MLflow/FastAPI/Streamlit architecture, adapted to the Kaggle Credit Card Fraud dataset.

## Key changes

- Target column: `Class`
- MLflow model name: `fraud-detection-automl`
- Experiment name: `credit-card-fraud-automl`
- Raw file location: `backend/data/raw/creditcard.csv`
- Training file location: `backend/data/processed/train.csv`

The canonical setup and repository instructions are in `README.md`.

## Dataset files

The full Kaggle dataset and generated train/test CSV files are kept locally and
are intentionally excluded from Git. Download `creditcard.csv` separately and
place it at `backend/data/raw/creditcard.csv`. The repository keeps only:

- `backend/data/sample_test.csv`
- `backend/data/sample_test_labeled.csv`

## Dataset summary

- Train rows: 227,845
- Test rows: 56,962
- Sample rows: 200
- Train class counts: {0: 227451, 1: 394}
- Sample class counts: {0: 180, 1: 20}

## Run

From the project root:

```powershell
.\start.ps1 -Rebuild
```

Then open:

- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000
- FastAPI docs: http://localhost:8000/docs

Upload:

- `backend\data\sample_test.csv` for predictions only
- `backend\data\sample_test_labeled.csv` for predictions + metrics/confusion matrix

## Regenerate processed data from raw

From the `backend` folder:

```powershell
cd backend
python preprocess_creditcard.py
```

This drops `Time`, fits `StandardScaler` on the training `Amount` only, transforms train/test, and creates the sample files.

## Configuration-driven experiments

All experiments use the single `backend/train.py` pipeline. From the `backend` directory, run:

```powershell
python train.py --config configs/baseline.yaml
```

Each YAML file defines the MLflow experiment name, H2O algorithm, model parameters,
feature version, and classification threshold. Every run logs the algorithm and
hyperparameters, feature version, threshold, accuracy, precision, recall, F1,
ROC-AUC, the serialized model, confusion matrix, and feature importance (when the
estimator exposes it). The completed model is registered under
`fraud-detection-automl@champion` and exported to the shared native-model volume,
so FastAPI and Streamlit continue to use the selected run.

### Experiment 1: Baseline model

```powershell
python train.py --config configs/baseline.yaml
```

Runs H2O AutoML with bounded runtime/model count and reproducible controls. Use
this as the reference for later experiments.

### Experiment 2: Hyperparameter tuning

```powershell
python train.py --config configs/gbm_tuned.yaml
```

Tests a GBM with explicit `ntrees`, `max_depth`, and `learn_rate` values. Copy the
config and change those values to create additional tracked runs without changing code.

### Experiment 3: Algorithm comparison

```powershell
python train.py --config configs/random_forest.yaml
```

Runs H2O Distributed Random Forest with configured `ntrees` and `max_depth`.
The model factory also supports `DRF` as an alias and `XGBoost` when the H2O
runtime has XGBoost support.

### Experiment 4: Feature engineering

```powershell
python train.py --config configs/feature_engineering.yaml
```

Enables `engineered_v1`, adding two interaction features and absolute scaled
amount. The feature version is stored with the deployment metadata, and FastAPI
applies the identical transformation during inference.

### Experiment 5: Threshold optimization

```powershell
python train.py --config configs/threshold.yaml
```

Evaluates and deploys the model with a `0.30` fraud-probability threshold instead
of `0.50`. Adjust the YAML threshold to compare the precision/recall tradeoff;
FastAPI reads the selected threshold from the saved model metadata.

Docker Compose runs `configs/baseline.yaml` by default. To deploy another experiment,
change the trainer's config path in `docker-compose.yml`, then rebuild and start the stack.
