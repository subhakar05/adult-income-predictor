"""
Streamlit app for the Adult/Census Income classifiers trained in lab.ipynb.

Run the notebook first (through the last cell) so that the `artifacts/`
folder exists next to this file, containing:
    sklearn_models.joblib, knn_data.npz, scaler.joblib, metadata.json,
    metrics.csv, overfitting.csv, confusion_matrices.png,
    roc_curves.png, metric_comparison.png

Then run:
    streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

NUMERIC_COLUMNS = [
    "age", "fnlwgt", "education-num",
    "capital-gain", "capital-loss", "hours-per-week"
]

# Same object definition used in the notebook - needed so predictions
# match exactly what was trained (Euclidean distance, majority vote of k).
class KNNFromScratch:
    def __init__(self, k=15):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def predict_proba(self, X):
        probabilities = []
        for row in np.asarray(X):
            distances = np.sum((self.X_train - row) ** 2, axis=1)
            nearest_rows = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_rows]
            probabilities.append(np.mean(nearest_labels == 1))
        return np.array(probabilities)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


@st.cache_resource
def load_artifacts():
    if not ARTIFACTS_DIR.exists():
        return None

    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text())
    scaler = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    sklearn_models = joblib.load(ARTIFACTS_DIR / "sklearn_models.joblib")

    knn_data = np.load(ARTIFACTS_DIR / "knn_data.npz")
    knn = KNNFromScratch(k=metadata["knn_k"])
    knn.X_train = knn_data["X_train"]
    knn.y_train = knn_data["y_train"]

    models = dict(sklearn_models)
    models["Custom KNN"] = knn

    metrics_path = ARTIFACTS_DIR / "metrics.csv"
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else None

    return {
        "metadata": metadata,
        "scaler": scaler,
        "models": models,
        "metrics_df": metrics_df,
    }


def build_feature_row(inputs: dict, feature_columns: list[str], categorical_columns: list[str]) -> pd.DataFrame:
    """
    Recreates the exact one-hot encoding used by pd.get_dummies(..., drop_first=True)
    at training time, for a single input row.
    """
    row = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

    for col in NUMERIC_COLUMNS:
        if col in row.columns:
            row.at[0, col] = inputs[col]

    for col in categorical_columns:
        dummy_col = f"{col}_{inputs[col]}"
        if dummy_col in row.columns:
            row.at[0, dummy_col] = 1
        # if not found, the selected value was the dropped reference
        # category (drop_first=True) - correctly stays all-zero for that group

    return row


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root {
            --ink: #252630; --muted: #69707d; --line: #d8d9df;
            --surface: #ffffff; --teal: #0e9f9b; --teal-soft: #e7f7f6;
            --coral: #ff5757;
        }
        html, body, [class*="css"] { font-size: 16px; }
        .stApp { background: #eef1f4; color: var(--ink); font-family: 'DM Sans', sans-serif; }
        .stApp p, .stApp label, .stApp [data-testid="stCaptionContainer"] { color: var(--ink); }
        [data-testid="stHeader"] { background: #ffffff !important; }
        [data-testid="stAppViewContainer"] > .main { padding-top: 1.3rem; }
        [data-testid="stAppViewContainer"] .block-container {
            max-width: 1400px; padding: 1.4rem 1.5rem 2rem;
            border: 1px solid #d8d9df; border-top: 3px solid var(--teal);
            border-radius: 16px; background: #ffffff;
            box-shadow: 0 12px 30px rgba(37, 38, 48, 0.1);
        }
        h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }

        .hero {
            padding: 1.1rem 1.5rem; border-radius: 8px; background: #ffffff;
            border: 1px solid var(--line); border-top: 0;
            box-shadow: none; margin-bottom: 1rem;
        }
        .hero h1 { margin: 0 0 0.25rem 0; font-size: 1.75rem; letter-spacing: -0.02em; }
        .hero p { margin: 0; color: var(--muted); font-size: 0.88rem; line-height: 1.4; }
        .hero code { color: var(--teal); background: var(--teal-soft); }

        .section-card {
            background: #ffffff; border: 1px solid var(--line);
            border-radius: 8px; padding: 1rem 1.1rem 0.45rem 1.1rem;
            margin-bottom: 0.8rem; box-shadow: 0 5px 18px rgba(37, 38, 48, 0.06);
        }
        .section-card h4 {
            margin: 0 0 0.9rem 0;
            font-size: 1rem; letter-spacing: 0.01em; color: var(--teal);
        }

        /* Form controls */
        label, .stMarkdown p { font-size: 0.92rem !important; }
        label { color: #252630 !important; font-weight: 600 !important; }
        div[data-baseweb="select"] * { font-size: 0.95rem !important; }
        input { font-size: 0.95rem !important; }
        .stNumberInput input, .stSelectbox div { font-size: 0.95rem !important; }
        div[data-baseweb="select"], div[data-baseweb="select"] > div,
        div[data-baseweb="select"] > div > div, div[role="combobox"],
        .stNumberInput, .stNumberInput > div, .stNumberInput input {
            border-color: #c9cdd5 !important; border-radius: 8px !important; background: #ffffff !important;
        }
        div[data-baseweb="select"] > div > div > div,
        [data-testid="stNumberInput"] > div > div,
        [data-testid="stNumberInput"] button {
            background: #ffffff !important; color: #555b68 !important;
        }
        div[data-baseweb="select"] *:not(svg):not(path) {
            background: #ffffff !important; background-color: #ffffff !important; color: #252630 !important;
        }
        div[data-baseweb="select"] svg, div[data-baseweb="select"] path { background: transparent !important; fill: #555b68 !important; color: #555b68 !important; }
        div[data-baseweb="select"] span, div[data-baseweb="select"] input,
        div[role="combobox"], .stNumberInput input { color: #252630 !important; }
        div[data-baseweb="select"] svg { fill: #555b68 !important; }
        div[data-baseweb="select"] > div, .stNumberInput > div { min-height: 38px; }
        .stNumberInput button { color: #555b68 !important; background: #ffffff !important; }
        .stNumberInput button:hover { color: #176b69 !important; background: #f3f4f6 !important; }
        [data-testid="stHorizontalBlock"] { gap: 0.8rem; }
        [data-testid="stWidgetLabel"] { margin-bottom: 0.15rem; }
        [data-testid="stNumberInput"], [data-testid="stSelectbox"] { margin-bottom: -0.3rem; }
        div[role="listbox"] { background: #ffffff !important; color: #252630 !important; border: 1px solid #c9cdd5; }
        div[role="listbox"] *, div[role="option"] { color: #252630 !important; background: #ffffff !important; }
        div[role="option"]:hover, div[role="option"][aria-selected="true"] { background: #f3f4f6 !important; color: #252630 !important; }

        div[data-testid="stFormSubmitButton"] button {
            width: auto;
            background: var(--coral); color: white;
            font-weight: 700;
            border: none;
            border-radius: 8px; padding: 0.7rem 1.1rem; font-size: 1rem;
        }
        div[data-testid="stFormSubmitButton"] button:hover { background: #e84949; }

        .result-card {
            min-height: 9rem; border-radius: 8px; padding: 1rem 1.1rem;
            border: 1px solid var(--line); background: var(--surface);
            margin-bottom: 0.7rem; box-shadow: 0 5px 18px rgba(37, 38, 48, 0.08);
        }
        .result-card .model-name { font-weight: 700; font-size: 0.95rem; margin-bottom: 0.7rem; color: var(--muted); }
        .result-card .prediction { font-family: 'Space Grotesk', sans-serif; font-size: 1.55rem; font-weight: 700; line-height: 1.1; margin-bottom: 0.45rem; color: var(--ink); }
        .result-card .prediction.badge-high { color: var(--teal); }
        .result-card .prediction.badge-low { color: #b44332; }
        .result-card .probability { font-size: 0.9rem; color: var(--muted); }
        .badge {
            display: inline-block;
            padding: 0.25rem 0.65rem; border-radius: 4px; font-size: 0.82rem;
            font-weight: 700;
        }
        .badge-high { background: #e5f3f0; color: var(--teal); }
        .badge-low { background: #fff0ed; color: #b44332; }

        .fdesc { color: var(--muted); font-size: 0.92rem; margin-bottom: 0.4rem; }
        .frange {
            color: var(--coral);
            font-family: monospace;
            font-size: 0.82rem;
        }
        [data-testid="stExpander"] { border-color: var(--line); background: #f7f7f8; }
        [data-testid="stProgressBar"] > div > div { background-color: var(--teal); }
        [data-testid="stWarning"] { border-left-color: var(--coral); background: #fff3ef; color: #7d2d22; }
        .guide { overflow: hidden; border: 1px solid var(--line); border-radius: 8px; background: #ffffff; margin-bottom: 1.6rem; }
        .guide-intro { padding: 1rem 1.2rem; color: var(--muted); font-size: 0.92rem; }
        .guide table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
        .guide th { background: var(--teal-soft); color: #176b69; text-align: left; font-weight: 600; }
        .guide th, .guide td { padding: 0.48rem 0.7rem; border: 1px solid #e1e3e8; }
        .guide td { color: #4d5360; }
        .guide td:first-child { color: #252630; font-weight: 600; }
        .guide td:last-child { color: var(--coral); }
        [data-testid="stForm"] { padding-bottom: 0.1rem; }
        [data-testid="stFormSubmitButton"] { margin-top: 0.35rem; }
        [data-testid="stProgressBar"] { margin-top: 0.3rem; }
        div[data-testid="stTabs"] { margin-top: 0.5rem; padding: 0.75rem; border: 1px solid var(--line); border-radius: 8px; background: rgba(255,255,255,0.85); }
        div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--teal); }
        @media (max-width: 700px) {
            [data-testid="stAppViewContainer"] .block-container { padding: 1rem 0.75rem 1.5rem; border-left: 0; border-right: 0; }
            .hero { padding: 1.35rem 1.1rem; }
            .hero h1 { font-size: 1.7rem; }
            .guide { overflow-x: auto; }
            .guide table { min-width: 680px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


NUMERIC_FEATURE_INFO = {
    "age": ("Age of the individual in years.", "17 – 90"),
    "fnlwgt": ("Census 'final weight' — how many people the record represents in the population.", "0 and up"),
    "education-num": ("Years of education, as a numeric code (higher = more schooling).", "1 – 16"),
    "capital-gain": ("Income from investment sources other than wages/salary.", "0 and up"),
    "capital-loss": ("Losses from investment sources other than wages/salary.", "0 and up"),
    "hours-per-week": ("Hours worked per week.", "1 – 99"),
}

CATEGORICAL_FEATURE_DESCRIPTIONS = {
    "workclass": "Type of employer (private, government, self-employed, etc.).",
    "education": "Highest education level completed.",
    "marital-status": "Current marital status.",
    "occupation": "General type of job.",
    "relationship": "Relationship role within the household.",
    "race": "Race category, as recorded in the census data.",
    "sex": "Sex, as recorded in the census data.",
    "native-country": "Country of origin.",
}


def render_feature_reference(categorical_options: dict):
    rows = []
    for col, (desc, rng) in NUMERIC_FEATURE_INFO.items():
        rows.append((col.replace("-", " ").title(), desc, rng))
    for col, options in categorical_options.items():
        desc = CATEGORICAL_FEATURE_DESCRIPTIONS.get(col, "")
        rng = ", ".join(str(o) for o in options)
        rows.append((col.replace("-", " ").title(), desc, rng))

    table_rows = "".join(
        f"<tr><td>{name}</td><td>{meaning}</td><td>{valid}</td></tr>"
        for name, meaning, valid in rows
    )
    with st.expander("ℹ️ Variable guide and valid values", expanded=False):
        st.markdown(
            f'''<div class="guide"><div class="guide-intro">Use values within the ranges used by the Adult dataset. Categorical fields accept the listed dataset categories.</div>
            <table><thead><tr><th>Variable</th><th>Meaning</th><th>Valid range or values</th></tr></thead>
            <tbody>{table_rows}</tbody></table></div>''',
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(page_title="Income Classifier", page_icon="💰", layout="wide")
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <h1>💰 Adult Income Prediction</h1>
            <p>Enter applicant details and compare predictions across every model trained in 
            — Decision Tree, SVM, Naive Bayes, MLP, and a from-scratch KNN.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data = load_artifacts()
    if data is None:
        st.error(
            f"No `artifacts/` folder found next to app.py at:\n\n`{ARTIFACTS_DIR}`\n\n"
            "Run every cell of lab.ipynb first (through the export cell) so the "
            "artifacts folder is created, then place it next to this app.py."
        )
        st.stop()

    metadata = data["metadata"]
    feature_columns = metadata["feature_columns"]
    categorical_options = metadata["categorical_options"]
    categorical_columns = list(categorical_options.keys())
    models = data["models"]

    render_feature_reference(categorical_options)

    form_col, sidebar_col = st.columns([2.4, 1], gap="large")

    with form_col:
        with st.form("input_form"):
            row1_left, row1_right = st.columns(2)
            with row1_left:
                age = st.number_input("Age", min_value=17, max_value=90, value=35)
            with row1_right:
                workclass = st.selectbox("Workclass", categorical_options["workclass"])

            row2_left, row2_right = st.columns(2)
            with row2_left:
                fnlwgt = st.number_input("Final weight (fnlwgt)", min_value=0, value=189778)
            with row2_right:
                education = st.selectbox("Education", categorical_options["education"])

            row3_left, row3_right = st.columns(2)
            with row3_left:
                education_num = st.number_input("Education-num", min_value=1, max_value=16, value=10)
            with row3_right:
                marital_status = st.selectbox("Marital status", categorical_options["marital-status"])

            row4_left, row4_right = st.columns(2)
            with row4_left:
                occupation = st.selectbox("Occupation", categorical_options["occupation"])
            with row4_right:
                relationship = st.selectbox("Relationship", categorical_options["relationship"])

            row5_left, row5_right = st.columns(2)
            with row5_left:
                race = st.selectbox("Race", categorical_options["race"])
            with row5_right:
                sex = st.selectbox("Sex", categorical_options["sex"])

            row6_left, row6_right = st.columns(2)
            with row6_left:
                capital_gain = st.number_input("Capital gain", min_value=0, value=0)
            with row6_right:
                capital_loss = st.number_input("Capital loss", min_value=0, value=0)

            row7_left, row7_right = st.columns(2)
            with row7_left:
                hours_per_week = st.number_input("Hours per week", min_value=1, max_value=99, value=40)
            with row7_right:
                native_country = st.selectbox("Native country", categorical_options["native-country"])

            submitted = st.form_submit_button("🔮 Predict Income")

    with sidebar_col:
        st.markdown('<div class="section-card"><h4>⚙️ Models to run</h4>', unsafe_allow_html=True)
        model_selection = st.selectbox(
            "Models",
            options=["All models", *models.keys()],
            index=0,
            label_visibility="collapsed",
        )
        model_choice = list(models.keys()) if model_selection == "All models" else [model_selection]
        st.caption("Choose all models or run one model at a time.")
        st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        return

    if not model_choice:
        st.warning("Select at least one model.")
        return

    inputs = {
        "age": age,
        "fnlwgt": fnlwgt,
        "education-num": education_num,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "workclass": workclass,
        "education": education,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "native-country": native_country,
    }

    raw_row = build_feature_row(inputs, feature_columns, categorical_columns)
    scaled_row = data["scaler"].transform(raw_row)

    results = []
    for name in model_choice:
        model = models[name]
        pred = model.predict(scaled_row)[0]

        proba_out = model.predict_proba(scaled_row)
        if name == "Custom KNN":
            prob_gt50k = float(proba_out[0])
        else:
            proba_out = np.asarray(proba_out)
            prob_gt50k = float(proba_out[0, 1]) if proba_out.ndim == 2 else float(proba_out[0])

        results.append({
            "Model": name,
            "Prediction": ">50K" if pred == 1 else "<=50K",
            "P(>50K)": round(prob_gt50k, 4),
        })

    results_df = pd.DataFrame(results)

    st.markdown("---")
    st.markdown("### 🎯 Results")

    card_cols = st.columns(len(results))
    for col, row in zip(card_cols, results):
        is_high = row["Prediction"] == ">50K"
        badge_class = "badge-high" if is_high else "badge-low"
        with col:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="model-name">{row['Model']}</div>
                    <div class="prediction {badge_class}">{row['Prediction']}</div>
                    <div class="probability">Probability: {row['P(>50K)']:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(row["P(>50K)"], 0.0), 1.0))
            st.caption(f"P(>50K) = {row['P(>50K)']:.4f}")

    tab1, tab2 = st.tabs(["📊 Comparison chart", "📋 Raw table"])
    with tab1:
        st.bar_chart(results_df.set_index("Model")["P(>50K)"])
    with tab2:
        st.dataframe(results_df, hide_index=True, use_container_width=True)

    if data["metrics_df"] is not None:
        with st.expander("📈 Model performance on the held-out test set"):
            st.dataframe(data["metrics_df"], hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()