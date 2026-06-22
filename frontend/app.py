# =========================================
# Streamlit UI for Project 3: Credit Card Fraud Detection
# H2O AutoML + MLflow + FastAPI + Streamlit
# =========================================
# Run locally: streamlit run app.py
import io
import json
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="AutoML Credit Card Fraud Detection", page_icon="CC", layout="wide")
st.title('Project 3: Credit Card Fraud Detection')
st.caption('H2O AutoML + MLflow + FastAPI + Streamlit')

ENDPOINT = os.getenv('BACKEND_URL', 'http://backend:8000/predict')

TARGET_COL = 'Class'
LABELS = {1: 'Fraud', 0: 'Legitimate'}

with st.expander('About this project and dataset', expanded=True):
    st.markdown(
        """
**Goal.** This project detects potentially fraudulent credit-card transactions using
an end-to-end MLOps architecture: H2O AutoML for model training, MLflow for tracking
and model registry, FastAPI for serving predictions, and Streamlit for the user interface.

**Dataset.** Credit Card Fraud Detection dataset. Each row is one transaction.
The original `Time` column was dropped and `Amount` was scaled. The model uses:

| Feature | Meaning |
|---|---|
| `V1` ... `V28` | PCA-transformed transaction features from the original dataset |
| `Amount` | Transaction amount, scaled during preprocessing |
| **`Class`** (target) | **1 = fraud**, **0 = legitimate** |

**Two ways to use this app:**
- Upload `sample_test.csv` -> predictions only.
- Upload `sample_test_labeled.csv` -> predictions plus accuracy, precision, recall, F1, and confusion matrix.
        """
    )

st.write(
    "Upload a processed credit-card transaction CSV, such as "
    "`backend/data/sample_test.csv` or `backend/data/sample_test_labeled.csv`, "
    "then click **Start Prediction**."
)

test_csv = st.file_uploader('Upload test dataset (CSV)', type=['csv'], accept_multiple_files=False)


def compute_metrics(y_true, y_pred):
    """Confusion-matrix counts and standard classification metrics without sklearn."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return tp, tn, fp, fn, accuracy, precision, recall, f1


if test_csv:
    test_df = pd.read_csv(test_csv)
    st.subheader('Sample of Uploaded Dataset')
    st.write(test_df.head())
    has_labels = TARGET_COL in test_df.columns
    st.write(
        f"Rows: {len(test_df)} | Columns: {test_df.shape[1]} | "
        f"Ground-truth labels: {'yes (evaluation enabled)' if has_labels else 'no (prediction only)'}"
    )

    test_bytes_obj = io.BytesIO()
    test_df.to_csv(test_bytes_obj, index=False)
    test_bytes_obj.seek(0)
    files = {"file": ('test_dataset.csv', test_bytes_obj, "multipart/form-data")}

    if st.button('Start Prediction'):
        if len(test_df) == 0:
            st.warning("Please upload a non-empty test dataset!")
        else:
            try:
                with st.spinner('Prediction in progress. Please wait...'):
                    response = requests.post(ENDPOINT, files=files, timeout=8000)
                response.raise_for_status()
                result = response.json()

                if isinstance(result, dict) and 'predictions' in result:
                    rows = result['predictions']
                    id_label = result.get('id_column', 'Transaction #')
                    results_df = pd.DataFrame(
                        {
                            id_label: [row.get('id') for row in rows],
                            'Prediction': [row.get('prediction') for row in rows],
                            'Fraud Probability': [row.get('probability') for row in rows],
                        }
                    )
                elif isinstance(result, dict):
                    results_df = pd.DataFrame({'Transaction ID': list(result.keys()), 'Prediction': list(result.values())})
                else:
                    results_df = pd.DataFrame({'Transaction #': range(1, len(result) + 1), 'Prediction': result})

                results_df['Prediction'] = results_df['Prediction'].astype(int)
                results_df['Result'] = results_df['Prediction'].map(lambda v: LABELS.get(v, str(v)))

                n_total = len(results_df)
                n_fraud = int((results_df['Prediction'] == 1).sum())
                n_legit = n_total - n_fraud
                pct = (n_fraud / n_total * 100) if n_total else 0

                st.success(f'Done! {n_total} transactions analysed.')

                st.subheader('Summary')
                c1, c2, c3 = st.columns(3)
                c1.metric('Transactions analysed', n_total)
                c2.metric('Predicted fraud', n_fraud, f'{pct:.1f}%')
                c3.metric('Predicted legitimate', n_legit)
                if 'Fraud Probability' in results_df.columns:
                    avg_prob = results_df['Fraud Probability'].dropna().mean()
                    st.metric('Average fraud probability', f'{avg_prob:.1%}' if pd.notna(avg_prob) else 'N/A')
                st.caption(
                    f"The model predicts that {n_fraud} of {n_total} transactions "
                    f"({pct:.1f}%) are potentially fraudulent and should be reviewed."
                )
                st.bar_chart(results_df['Result'].value_counts(), use_container_width=True)

                if has_labels:
                    y_true = test_df[TARGET_COL].astype(int).tolist()
                    y_pred = results_df['Prediction'].tolist()
                    tp, tn, fp, fn, acc, prec, rec, f1 = compute_metrics(y_true, y_pred)

                    st.subheader('Model Evaluation')
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric('Accuracy', f'{acc:.1%}')
                    m2.metric('Precision', f'{prec:.1%}')
                    m3.metric('Recall', f'{rec:.1%}')
                    m4.metric('F1-score', f'{f1:.1%}')

                    st.markdown('**Confusion Matrix** (rows = actual, columns = predicted)')
                    cm = pd.DataFrame(
                        [[tn, fp], [fn, tp]],
                        index=['Actual: Legitimate', 'Actual: Fraud'],
                        columns=['Predicted: Legitimate', 'Predicted: Fraud'],
                    )
                    st.table(cm)

                    st.markdown(
                        f"""
**How to read this:**
- **True Negatives ({tn})** - legitimate transactions correctly identified as legitimate.
- **True Positives ({tp})** - fraudulent transactions correctly identified as fraud.
- **False Positives ({fp})** - legitimate transactions incorrectly flagged as fraud.
- **False Negatives ({fn})** - fraudulent transactions missed by the model.

For fraud detection, **recall** is especially important because false negatives mean missed fraud.
Precision is also important because false positives create unnecessary manual reviews.
                        """
                    )

                st.subheader('Detailed results')
                show_cols = [results_df.columns[0], 'Result']
                if 'Fraud Probability' in results_df.columns:
                    results_df['Fraud Probability'] = results_df['Fraud Probability'].map(
                        lambda v: f'{v:.1%}' if pd.notna(v) else 'N/A'
                    )
                    show_cols.append('Fraud Probability')
                if has_labels:
                    results_df['Actual'] = [LABELS.get(int(v), str(v)) for v in test_df[TARGET_COL]]
                    results_df['Correct'] = ['yes' if int(a) == int(p) else 'no' for a, p in zip(test_df[TARGET_COL], results_df['Prediction'])]
                    show_cols += ['Actual', 'Correct']
                st.dataframe(results_df[show_cols], use_container_width=True, hide_index=True)

                d1, d2 = st.columns(2)
                d1.download_button(
                    label='Download results (CSV)',
                    data=results_df.to_csv(index=False),
                    file_name='fraud_prediction_results.csv',
                    mime='text/csv',
                )
                d2.download_button(
                    label='Download raw (JSON)',
                    data=json.dumps(result),
                    file_name='automl_fraud_prediction_results.json',
                    mime='application/json',
                )
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach the prediction backend at {ENDPOINT}.")
                st.exception(exc)
