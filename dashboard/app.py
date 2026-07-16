import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path

st.set_page_config(
    page_title="AI Women's Safety Decision Intelligence Platform",
    page_icon="🛡️",
    layout="wide"
)

LOG_PATH = Path("output/realtime_predictions_log.csv")
FEATURE_PATH = Path("output/latest_features.csv")
MODEL_PATH = Path("models/xgb_risk_level.pkl")

st.title("🛡️ AI Spatio-Temporal Women's Safety Decision Intelligence Platform")

if not LOG_PATH.exists():

    st.warning("Run real_time_loop.py first.")

    st.stop()

df = pd.read_csv(LOG_PATH)

if df.empty:

    st.warning("Prediction log is empty.")

    st.stop()

# ==============================
# KPI
# ==============================

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Events", len(df))

c2.metric(
    "High Risk",
    (df["predicted_risk_level"]=="high").sum()
)

c3.metric(
    "Average High Risk Probability",
    f"{df['risk_probability'].mean():.2%}"
)

c4.metric(
    "Latest Prediction",
    df.iloc[-1]["predicted_risk_level"].upper()
)

st.divider()

# ==============================
# Charts
# ==============================

st.subheader("Risk Distribution")

st.bar_chart(df["predicted_risk_level"].value_counts())

st.subheader("Area Distribution")

st.bar_chart(df["area_type"].value_counts())

st.subheader("Purpose Distribution")

st.bar_chart(df["purpose"].value_counts())

st.divider()

# ==============================
# Recent Predictions
# ==============================

st.subheader("Recent Predictions")

st.dataframe(
    df.tail(20),
    use_container_width=True
)

st.divider()

# ==============================
# SHAP Explainability
# ==============================

st.subheader("🧠 Explainable AI (SHAP)")

if FEATURE_PATH.exists():

    try:

        X = pd.read_csv(FEATURE_PATH)

        model = joblib.load(MODEL_PATH)

        explainer = shap.Explainer(model)

        shap_values = explainer(X)

        st.write("Top Features Influencing Current Prediction")

        fig, ax = plt.subplots(figsize=(10,5))

        shap.plots.bar(
            shap_values[:, :, 0],
            max_display=10,
            show=False
        )

        st.pyplot(fig)

    except Exception as e:

        st.warning(f"SHAP Error: {e}")

else:

    st.info("Run real_time_loop.py first.")