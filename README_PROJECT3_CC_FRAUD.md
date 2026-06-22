# Project 3 - Credit Card Fraud Detection with H2O AutoML + MLflow

This is a complete drop-in project folder for the teacher H2O/MLflow/FastAPI/Streamlit architecture, adapted to the Kaggle Credit Card Fraud dataset.

## Key changes

- Target column: `Class`
- MLflow model name: `fraud-detection-automl`
- Experiment name: `credit-card-fraud-automl`
- Raw file location: `backend/data/raw/creditcard.csv`
- Training file location: `backend/data/processed/train.csv`

## Included data files

- `backend/data/raw/creditcard.csv`
- `backend/data/processed/train.csv`
- `backend/data/processed/test.csv`
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

## Dataset Setup

The full Kaggle Credit Card Fraud dataset is not included in this repository because of file size limits.

Download the dataset from Kaggle https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place:


backend/data/raw/creditcard.csv

Then run:

notebooks/01_EDA_and_Data_PreProcessing_CC_Fraud.ipynb

This generates:

backend/data/processed/train.csv
backend/data/processed/test.csv
backend/data/sample_test.csv
backend/data/sample_test_labeled.csv
