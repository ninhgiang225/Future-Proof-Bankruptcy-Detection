import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.ui_helpers import (
    COLORS,
    FEATURE_CATEGORIES,
    get_shap_explainer,
    compute_shap_values,
    risk_gauge,
    shap_waterfall_plot,
    shap_top_contributors,
    global_feature_importance,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bankruptcy Risk Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("Bankruptcy Risk Prediction")
st.markdown(
    "Predict the likelihood of corporate bankruptcy using financial indicators "
    "and understand **which features drive the prediction**."
)

# ---------------------------------------------------------------------------
# Load model & data (cached)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "models" / "bankruptcy_prediction_model.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "data" / "data.csv")
    df.columns = df.columns.str.strip()
    return df


model = load_model()
df = load_data()
feature_names = [c for c in df.columns if c != "Bankrupt?"]
medians = df[feature_names].median()

# ---------------------------------------------------------------------------
# Sidebar — input mode selector
# ---------------------------------------------------------------------------
st.sidebar.header("Input Financial Metrics")
input_mode = st.sidebar.radio(
    "Choose input method",
    ["Manual Entry", "CSV Upload", "Sample Company"],
)

inputs: dict[str, float] = {}

# ---- Manual Entry ---------------------------------------------------------
if input_mode == "Manual Entry":
    st.sidebar.markdown("*Fields are pre-filled with dataset medians.*")
    for group_name, features in FEATURE_CATEGORIES.items():
        with st.sidebar.expander(group_name, expanded=False):
            for feat in features:
                if feat in feature_names:
                    inputs[feat] = st.number_input(
                        feat,
                        value=float(medians[feat]),
                        format="%.6f",
                        key=f"manual_{feat}",
                    )
    # fill any features not listed in categories (safety net)
    for feat in feature_names:
        if feat not in inputs:
            inputs[feat] = float(medians[feat])

# ---- CSV Upload -----------------------------------------------------------
elif input_mode == "CSV Upload":
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        udf = pd.read_csv(uploaded)
        udf.columns = udf.columns.str.strip()
        missing = set(feature_names) - set(udf.columns)
        if missing:
            st.sidebar.error(f"Missing columns: {missing}")
        else:
            row_idx = 0
            if len(udf) > 1:
                row_idx = st.sidebar.selectbox(
                    "Select row", range(len(udf)), format_func=lambda i: f"Row {i}"
                )
            for feat in feature_names:
                inputs[feat] = float(udf.iloc[row_idx][feat])
            st.sidebar.success(f"Loaded row {row_idx} ({len(udf)} rows in file)")
    if not inputs:
        for feat in feature_names:
            inputs[feat] = float(medians[feat])

# ---- Sample Company -------------------------------------------------------
else:
    sample_idx = st.sidebar.selectbox(
        "Pick a company from the dataset",
        range(len(df)),
        format_func=lambda i: f"Company {i} ({'Bankrupt' if df.iloc[i]['Bankrupt?'] == 1 else 'Non-Bankrupt'})",
    )
    for feat in feature_names:
        inputs[feat] = float(df.iloc[sample_idx][feat])
    st.sidebar.info(f"Loaded company {sample_idx}")

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
predict_clicked = st.sidebar.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    input_values = np.array([inputs[f] for f in feature_names]).reshape(1, -1)
    input_df = pd.DataFrame(input_values, columns=feature_names)

    proba = model.predict_proba(input_values)[0]
    pred_class = model.predict(input_values)[0]
    bankruptcy_prob = proba[1]

    # --- Risk banner -------------------------------------------------------
    if bankruptcy_prob < 0.3:
        bg, border, text_col, label = (
            COLORS["safe_bg"], COLORS["safe_border"], COLORS["safe_border"], "LOW RISK"
        )
    elif bankruptcy_prob < 0.7:
        bg, border, text_col, label = (
            COLORS["warn_bg"], COLORS["warn_border"], COLORS["warn_border"], "MEDIUM RISK"
        )
    else:
        bg, border, text_col, label = (
            COLORS["danger_bg"], COLORS["danger_border"], COLORS["danger_border"], "HIGH RISK"
        )

    st.markdown(
        f'<div style="background:{bg};border-left:6px solid {border};'
        f'padding:16px 24px;border-radius:6px;color:{text_col};'
        f'text-align:center;font-size:1.4rem;font-weight:700;margin-bottom:1rem;">'
        f'{label}</div>',
        unsafe_allow_html=True,
    )

    # --- Metric cards ------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Prediction", "Bankrupt" if pred_class == 1 else "Non-Bankrupt")
    col2.metric("Bankruptcy Probability", f"{bankruptcy_prob:.2%}")
    col3.metric("Confidence", f"{max(proba):.2%}")

    # --- Gauge chart -------------------------------------------------------
    st.plotly_chart(risk_gauge(bankruptcy_prob), use_container_width=True)

    # --- Explainability tabs -----------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "Why this prediction?",
        "Top Contributors",
        "Global Feature Importance",
    ])

    with tab1:
        with st.spinner("Computing SHAP values..."):
            explainer = get_shap_explainer(model)
            shap_vals = compute_shap_values(explainer, input_df)
        st.plotly_chart(
            shap_waterfall_plot(shap_vals, feature_names, top_n=15),
            use_container_width=True,
        )

    with tab2:
        with st.spinner("Computing SHAP values..."):
            explainer = get_shap_explainer(model)
            shap_vals = compute_shap_values(explainer, input_df)
        fig, contrib_df = shap_top_contributors(shap_vals, feature_names, top_n=10)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    with tab3:
        st.plotly_chart(
            global_feature_importance(model, feature_names, top_n=20),
            use_container_width=True,
        )
