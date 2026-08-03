"""Streamlit interface for the credit-card fraud detection service."""

import io
import json
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st


st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ENDPOINT = os.getenv("BACKEND_URL", "http://backend:8000/predict")
TARGET_COL = "Class"
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABELS = {1: "Fraud", 0: "Legitimate"}


def apply_theme():
    st.markdown(
        """
        <style>
        :root {
            --bg: #081020;
            --panel: rgba(12, 28, 54, .78);
            --line: rgba(45, 211, 255, .28);
            --line-strong: rgba(45, 211, 255, .62);
            --cyan: #22d3ee;
            --blue: #3b82f6;
            --purple: #7c3aed;
            --green: #10b981;
            --amber: #f59e0b;
            --text: #e5f2ff;
            --muted: #9fb3c8;
        }
        .stApp {
            background:
                radial-gradient(circle at 18% 0%, rgba(34,211,238,.12), transparent 28%),
                radial-gradient(circle at 82% 10%, rgba(124,58,237,.13), transparent 30%),
                linear-gradient(135deg, #050914 0%, #081020 52%, #03101d 100%);
            color: var(--text);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(3,10,24,.98), rgba(8,18,36,.96));
            border-right: 1px solid rgba(45,211,255,.2);
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label {
            border: 1px solid rgba(45,211,255,.12);
            border-radius: 12px;
            padding: .55rem .75rem;
            margin-bottom: .35rem;
            background: rgba(12,28,54,.45);
            transition: 160ms ease;
        }
        section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
            border-color: var(--line-strong);
            box-shadow: 0 0 18px rgba(34,211,238,.16);
            transform: translateX(2px);
        }
        .block-container { padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1240px; }
        .sidebar-brand, .sidebar-tagline {
            border: 1px solid rgba(45,212,255,.22);
            background: linear-gradient(145deg, rgba(15,37,66,.92), rgba(5,15,31,.82));
            box-shadow: 0 14px 36px rgba(0,0,0,.18);
        }
        .sidebar-brand { border-radius: 16px; padding: 1rem; margin: .4rem 0 1rem; }
        .sidebar-brand-title { color: var(--text); font-size: 1.08rem; font-weight: 800; margin: 0; }
        .sidebar-brand-subtitle { color: var(--cyan); font-size: .82rem; margin: .2rem 0 0; }
        .sidebar-tagline { border-radius: 16px; padding: .9rem; margin-top: 1.25rem; color: var(--muted); }
        .sidebar-tagline strong { color: var(--cyan); }
        .hero {
            border: 1px solid var(--line); border-radius: 18px; padding: 1.7rem 1.9rem;
            margin-bottom: 1rem; min-height: 178px; position: relative; overflow: hidden;
            background:
                radial-gradient(circle at 80% 12%, rgba(34,211,238,.24), transparent 16%),
                radial-gradient(circle at 92% 52%, rgba(37,99,235,.28), transparent 22%),
                radial-gradient(circle at 76% 85%, rgba(124,58,237,.16), transparent 24%),
                linear-gradient(135deg, rgba(12,28,54,.95), rgba(5,15,31,.84));
            box-shadow: 0 18px 55px rgba(0,0,0,.28), 0 0 34px rgba(34,211,238,.18), inset 0 1px 0 rgba(255,255,255,.05);
        }
        .hero::after { content:""; position:absolute; right:-60px; bottom:-90px; width:260px; height:260px; border-radius:50%; border:1px solid rgba(34,211,238,.18); box-shadow:0 0 70px rgba(34,211,238,.2); }
        .hero-kicker { color: var(--cyan); font-size: .76rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
        .hero h1 { color: var(--text); margin: .25rem 0; font-size: clamp(2rem,4vw,3.05rem); line-height:1.05; max-width:68%; }
        .hero p { color: var(--muted); margin: .55rem 0 0; max-width: 68%; }
        .hero-art { position:absolute; right:1.35rem; top:1.2rem; width:34%; min-width:230px; height:calc(100% - 2.4rem); pointer-events:none; }
        .shield-icon { position:absolute; right:5.1rem; top:.2rem; width:116px; height:100px; border:1px solid rgba(34,211,238,.42); border-radius:28px; display:grid; place-items:center; font-size:3rem; background:radial-gradient(circle,rgba(34,211,238,.25),transparent 50%),linear-gradient(145deg,rgba(37,99,235,.42),rgba(3,10,24,.58)); box-shadow:0 0 34px rgba(34,211,238,.34),inset 0 0 18px rgba(34,211,238,.16); }
        .signal-chip { position:absolute; width:54px; height:46px; border:1px solid rgba(34,211,238,.32); border-radius:14px; display:grid; place-items:center; color:var(--cyan); background:rgba(3,10,24,.52); box-shadow:0 0 20px rgba(37,99,235,.22); font-size:1.35rem; }
        .signal-chip.one { right:.3rem; top:.15rem; } .signal-chip.two { right:1.8rem; bottom:.35rem; } .signal-chip.three { right:10.3rem; bottom:.8rem; }
        .dataset-banner { border:1px solid rgba(34,211,238,.35); border-radius:14px; padding:.82rem 1rem; margin:.75rem 0 1.1rem; color:#b9ecff; background:linear-gradient(90deg,rgba(14,116,144,.22),rgba(37,99,235,.12)); }
        .fintech-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.7rem; margin:1rem 0; }
        .kpi-card { min-height:138px; border:1px solid rgba(45,211,255,.24); border-radius:20px; padding:.85rem; background:radial-gradient(circle at 18% 18%,rgba(34,211,238,.13),transparent 24%),linear-gradient(145deg,rgba(12,28,54,.86),rgba(5,15,31,.78)); box-shadow:0 16px 42px rgba(0,0,0,.25),inset 0 1px 0 rgba(255,255,255,.04); transition:160ms ease; }
        .kpi-card:hover { transform:translateY(-3px); border-color:rgba(34,211,238,.78); box-shadow:0 22px 56px rgba(0,0,0,.35),0 0 28px rgba(34,211,238,.22); }
        .kpi-top { display:flex; align-items:flex-start; gap:.65rem; margin-bottom:.7rem; }
        .kpi-icon { width:42px; height:42px; border-radius:14px; display:grid; place-items:center; font-size:1.22rem; background:linear-gradient(135deg,#0ea5e9,#2563eb); box-shadow:0 0 28px rgba(34,211,238,.24); flex:0 0 42px; }
        .kpi-icon-1 { background:linear-gradient(135deg,#7c3aed,#22d3ee); } .kpi-icon-2 { background:linear-gradient(135deg,#f59e0b,#22d3ee); }
        .kpi-label { color:var(--muted); font-size:.76rem; font-weight:700; } .kpi-value { color:#f8fbff; font-size:1.3rem; font-weight:900; } .kpi-note { color:var(--green); font-size:.75rem; margin-top:.55rem; }
        .pipeline-panel { border:1px solid rgba(45,211,255,.2); border-radius:18px; padding:1.05rem 1.2rem; margin:1rem 0; background:radial-gradient(circle at 90% 50%,rgba(37,99,235,.2),transparent 28%),linear-gradient(90deg,rgba(12,28,54,.78),rgba(5,15,31,.66)); box-shadow:0 0 34px rgba(34,211,238,.14),0 18px 48px rgba(0,0,0,.24); }
        .pipeline-title { color:var(--cyan); font-weight:900; letter-spacing:.06em; text-transform:uppercase; margin-bottom:.85rem; }
        .pipeline-strip { display:grid; grid-template-columns:1fr auto 1fr auto 1fr auto 1fr; gap:.55rem; align-items:center; }
        .pipeline-step { text-align:center; color:var(--muted); font-size:.8rem; } .pipeline-step span { display:block; color:var(--cyan); font-size:1.55rem; margin-bottom:.35rem; text-shadow:0 0 18px rgba(34,211,238,.38); } .pipeline-arrow { color:var(--cyan); font-size:1.35rem; }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(45,212,255,.18);
            border-radius: 15px;
            padding: .8rem 1rem;
            background: rgba(11,31,55,.76);
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px !important;
            border: 1px solid rgba(45,212,255,.42) !important;
            background: linear-gradient(135deg, rgba(37,99,235,.95), rgba(124,58,237,.9)) !important;
            color:white !important; box-shadow:0 0 22px rgba(37,99,235,.25); transition:160ms ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--cyan)!important; box-shadow:0 0 26px rgba(34,211,238,.32); transform:translateY(-1px); }
        div[data-testid="stDataFrame"], div[data-testid="stFileUploaderDropzone"], div[data-testid="stExpander"] { border:1px solid rgba(45,211,255,.16); border-radius:16px; background:rgba(5,15,31,.45); box-shadow:0 14px 44px rgba(0,0,0,.18); }
        div[data-testid="stFileUploaderDropzone"] { background:linear-gradient(145deg,rgba(12,28,54,.72),rgba(5,15,31,.64)); }
        div[data-baseweb="tab-list"] { gap:.45rem; } button[data-baseweb="tab"] { border-radius:12px 12px 0 0; color:var(--muted); }
        .stAlert { border-radius:14px; }
        @media(max-width:900px) { .hero h1,.hero p{max-width:100%}.hero-art{display:none}.pipeline-strip{grid-template-columns:1fr}.fintech-grid{grid-template-columns:1fr} }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(icon, title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">FraudGuard AI · MLOps workspace</div>
            <h1>{icon} {title}</h1>
            <p>{subtitle}</p>
            <div class="hero-art">
                <div class="shield-icon">🛡️</div>
                <div class="signal-chip one">◎</div>
                <div class="signal-chip two">↗</div>
                <div class="signal-chip three">▥</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(cards):
    html = ['<div class="fintech-grid">']
    for index, (icon, label, value, note) in enumerate(cards):
        html.append(
            f"""<div class="kpi-card"><div class="kpi-top">
            <div class="kpi-icon kpi-icon-{index}">{icon}</div><div>
            <div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>
            </div></div><div class="kpi-note">{note}</div></div>"""
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def generate_demo_datasets(rows=500, fraud_rate=0.08, seed=42):
    """Generate model-compatible synthetic PCA features with optional labels."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(rows) < fraud_rate).astype(int)
    features = rng.normal(0, 1, size=(rows, 28))

    # Add a visible fraud signal to a subset of PCA dimensions. The values remain
    # plausible for the anonymized feature space while producing a useful demo mix.
    fraud_rows = labels == 1
    signal = rng.normal(0, 0.7, size=(fraud_rows.sum(), 6))
    features[fraud_rows, :6] += signal + np.array([-2.8, 2.2, -3.0, 2.6, -1.8, -2.2])
    features[fraud_rows, 9:14] += np.array([-2.5, 1.6, -2.0, -1.5, -2.4])

    amounts = rng.lognormal(mean=-0.15, sigma=0.85, size=rows)
    amounts = (amounts - amounts.mean()) / amounts.std()
    labeled = pd.DataFrame(features, columns=FEATURE_COLS[:-1])
    labeled["Amount"] = amounts
    labeled[TARGET_COL] = labels
    return labeled, labeled.drop(columns=TARGET_COL)


def validate_dataset(df):
    missing = [column for column in FEATURE_COLS if column not in df.columns]
    unexpected = [column for column in df.columns if column not in FEATURE_COLS + [TARGET_COL]]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    if unexpected:
        return False, f"Unexpected columns: {', '.join(unexpected)}"
    if df.empty:
        return False, "The dataset contains no rows."
    if df[FEATURE_COLS].isna().any().any():
        return False, "Feature columns contain missing values."
    if not all(pd.api.types.is_numeric_dtype(df[column]) for column in FEATURE_COLS):
        return False, "All feature columns must be numeric."
    if TARGET_COL in df and not set(df[TARGET_COL].dropna().astype(int).unique()).issubset({0, 1}):
        return False, "Class must contain only 0 (legitimate) and 1 (fraud)."
    return True, ""


def compute_metrics(y_true, y_pred):
    tp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == predicted == 1)
    tn = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == predicted == 0)
    fp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == 0 and predicted == 1)
    fn = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == 1 and predicted == 0)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return tp, tn, fp, fn, accuracy, precision, recall, f1


def select_for_prediction(df, name):
    st.session_state["active_dataset"] = df.copy()
    st.session_state["active_dataset_name"] = name
    st.session_state["page"] = "🔍 Prediction"


def parse_prediction_response(result):
    if isinstance(result, dict) and "predictions" in result:
        rows = result["predictions"]
        id_label = result.get("id_column", "Transaction #")
        return pd.DataFrame(
            {
                id_label: [row.get("id") for row in rows],
                "Prediction": [row.get("prediction") for row in rows],
                "Fraud Probability": [row.get("probability") for row in rows],
            }
        )
    if isinstance(result, dict):
        return pd.DataFrame({"Transaction ID": list(result), "Prediction": list(result.values())})
    return pd.DataFrame({"Transaction #": range(1, len(result) + 1), "Prediction": result})


def render_prediction_results(test_df, result):
    results_df = parse_prediction_response(result)
    results_df["Prediction"] = results_df["Prediction"].astype(int)
    results_df["Result"] = results_df["Prediction"].map(LABELS)
    total = len(results_df)
    fraud = int((results_df["Prediction"] == 1).sum())
    has_labels = TARGET_COL in test_df

    st.success(f"Done — {total:,} transactions analysed.")
    render_kpi_cards(
        [
            ("▥", "Transactions", f"{total:,}", "Rows scored by the active champion model"),
            ("⚠", "Predicted Fraud", f"{fraud:,}", f"{fraud / total:.1%} of analysed transactions" if total else "No rows"),
            ("✓", "Predicted Legitimate", f"{total - fraud:,}", "Transactions below the fraud threshold"),
        ]
    )
    st.bar_chart(results_df["Result"].value_counts(), use_container_width=True)

    if has_labels:
        y_true = test_df[TARGET_COL].astype(int).tolist()
        y_pred = results_df["Prediction"].tolist()
        tp, tn, fp, fn, accuracy, precision, recall, f1 = compute_metrics(y_true, y_pred)
        st.subheader("Model evaluation")
        metrics = st.columns(4)
        for column, label, value in zip(metrics, ["Accuracy", "Precision", "Recall", "F1 score"], [accuracy, precision, recall, f1]):
            column.metric(label, f"{value:.1%}")
        st.table(
            pd.DataFrame(
                [[tn, fp], [fn, tp]],
                index=["Actual: Legitimate", "Actual: Fraud"],
                columns=["Predicted: Legitimate", "Predicted: Fraud"],
            )
        )
        results_df["Actual"] = [LABELS[int(value)] for value in test_df[TARGET_COL]]
        results_df["Correct"] = ["Yes" if int(a) == int(p) else "No" for a, p in zip(y_true, y_pred)]

    st.subheader("Detailed results")
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("Download results (CSV)", results_df.to_csv(index=False), "fraud_prediction_results.csv", "text/csv")
    d2.download_button("Download raw response (JSON)", json.dumps(result), "fraud_prediction_results.json", "application/json")


def render_prediction_page():
    page_header("🔍", "Transaction Prediction", "Upload a compatible CSV or use a generated demo dataset to score transactions.")
    st.markdown('<div class="dataset-banner">📤 Drop in a compatible CSV, or generate a ready-to-use labeled or unlabeled demo from the sidebar.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a labeled or unlabeled CSV", type=["csv"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            st.session_state["active_dataset"] = pd.read_csv(uploaded)
            st.session_state["active_dataset_name"] = uploaded.name
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not read this CSV: {exc}")
            return

    test_df = st.session_state.get("active_dataset")
    if test_df is None:
        st.info("Upload a CSV here, or create one on the Demo Dataset Generator page.")
        return

    valid, message = validate_dataset(test_df)
    if not valid:
        st.error(message)
        return
    name = st.session_state.get("active_dataset_name", "Dataset")
    label_status = "Labeled · evaluation enabled" if TARGET_COL in test_df else "Unlabeled · prediction only"
    st.subheader(name)
    st.caption(f"{len(test_df):,} rows · {test_df.shape[1]} columns · {label_status}")
    st.dataframe(test_df.head(12), use_container_width=True, hide_index=True)

    if st.button("Start Prediction", type="primary", use_container_width=True):
        payload = io.BytesIO(test_df.to_csv(index=False).encode("utf-8"))
        try:
            with st.spinner("Scoring transactions..."):
                response = requests.post(
                    ENDPOINT,
                    files={"file": ("test_dataset.csv", payload, "text/csv")},
                    timeout=8000,
                )
            response.raise_for_status()
            render_prediction_results(test_df, response.json())
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the prediction backend at {ENDPOINT}.")
            st.exception(exc)


def render_generator_page():
    page_header("🎲", "Demo Dataset Generator", "Create matching labeled and unlabeled transaction datasets, then download or load either one directly into Prediction.")
    left, middle, right = st.columns(3)
    rows = left.slider("Transactions", 50, 5000, 500, 50)
    fraud_rate = middle.slider("Fraud rate (fraction)", 0.01, 0.40, 0.08, 0.01, format="%.2f")
    seed = right.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1)

    if st.button("Generate labeled + unlabeled datasets", type="primary", use_container_width=True):
        labeled, unlabeled = generate_demo_datasets(rows, fraud_rate, int(seed))
        st.session_state["demo_labeled"] = labeled
        st.session_state["demo_unlabeled"] = unlabeled
        st.success(f"Created two compatible datasets with {rows:,} rows each.")

    labeled = st.session_state.get("demo_labeled")
    unlabeled = st.session_state.get("demo_unlabeled")
    if labeled is None or unlabeled is None:
        st.info("Choose the demo settings and generate both datasets.")
        return

    fraud_count = int(labeled[TARGET_COL].sum())
    render_kpi_cards(
        [
            ("▥", "Rows Per Dataset", f"{len(labeled):,}", "Same transactions in both generated files"),
            ("⚠", "Fraud Labels", f"{fraud_count:,}", "Positive examples available for evaluation"),
            ("✓", "Legitimate Labels", f"{len(labeled) - fraud_count:,}", "Negative examples available for evaluation"),
        ]
    )

    labeled_tab, unlabeled_tab = st.tabs(["Labeled dataset", "Unlabeled dataset"])
    with labeled_tab:
        st.caption("Includes Class, so Prediction also calculates evaluation metrics.")
        st.dataframe(labeled.head(12), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button("Download labeled CSV", labeled.to_csv(index=False), "demo_fraud_labeled.csv", "text/csv", use_container_width=True)
        c2.button(
            "Use labeled dataset in Prediction",
            use_container_width=True,
            on_click=select_for_prediction,
            args=(labeled, "Generated labeled demo"),
        )
    with unlabeled_tab:
        st.caption("Omits Class, matching a real inference-only upload.")
        st.dataframe(unlabeled.head(12), use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button("Download unlabeled CSV", unlabeled.to_csv(index=False), "demo_fraud_unlabeled.csv", "text/csv", use_container_width=True)
        c2.button(
            "Use unlabeled dataset in Prediction",
            use_container_width=True,
            on_click=select_for_prediction,
            args=(unlabeled, "Generated unlabeled demo"),
        )


def render_about_page():
    page_header("🧭", "About This Project", "A concise guide to the data contract and end-to-end fraud detection workflow.")
    st.markdown(
        """
        <div class="pipeline-panel">
          <div class="pipeline-title">Live MLOps Pipeline</div>
          <div class="pipeline-strip">
            <div class="pipeline-step"><span>▥</span>CSV transactions</div><div class="pipeline-arrow">→</div>
            <div class="pipeline-step"><span>⚙</span>FastAPI service</div><div class="pipeline-arrow">→</div>
            <div class="pipeline-step"><span>◈</span>H2O champion</div><div class="pipeline-arrow">→</div>
            <div class="pipeline-step"><span>◎</span>Fraud insights</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        ### Architecture

        H2O AutoML trains the model, MLflow tracks and registers it, FastAPI serves
        predictions, and this Streamlit interface handles data preparation and review.

        ### Dataset contract

        Each row is one transaction. The model expects `V1` through `V28` plus a
        scaled `Amount`. A `Class` column is optional: `1` means fraud and `0` means
        legitimate. Labeled uploads enable accuracy, precision, recall, F1, and a
        confusion matrix; unlabeled uploads run prediction only.

        Generated data is synthetic and intended for demonstrations and integration
        testing—not for model training or performance claims.
        """
    )


def main():
    apply_theme()
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <p class="sidebar-brand-title">🛡️ FraudGuard AI</p>
            <p class="sidebar-brand-subtitle">Credit-card fraud intelligence</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pages = ["🔍 Prediction", "🎲 Demo Dataset Generator", "🧭 About"]
    if "page" not in st.session_state:
        st.session_state["page"] = pages[0]
    page = st.sidebar.radio("Navigation", pages, key="page")

    if page == pages[1]:
        render_generator_page()
    elif page == pages[2]:
        render_about_page()
    else:
        render_prediction_page()

    active_name = st.session_state.get("active_dataset_name")
    if active_name:
        st.sidebar.caption(f"Active dataset: {active_name}")
    st.sidebar.markdown(
        '<div class="sidebar-tagline">✨ <strong>From transactions</strong><br>to actionable fraud signals.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
