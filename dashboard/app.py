import pandas as pd
import streamlit as st
from pathlib import Path

LOG_PATH = Path("output/realtime_predictions_log.csv")

st.set_page_config(
    page_title="Spatio-Temporal Risk Dashboard",
    page_icon="⚠️",
    layout="wide",
)

st.title("Spatio-Temporal Risk Prediction Dashboard")
st.write("Real-time (simulated) safety risk predictions for girls in CCTV-blind areas.")

if not LOG_PATH.exists():
    st.warning("Log file not found. Run the real-time loop first to generate predictions.")
else:
    df = pd.read_csv(LOG_PATH)

    if df.empty:
        st.info("Log file is empty. Let the real-time loop run for a while, then refresh.")
    else:
        # Show some summary stats
        st.subheader("Current Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Events Logged", len(df))

        with col2:
            high_count = (df["predicted_risk_level"] == "high").sum()
            st.metric("High Risk Events", int(high_count))

        with col3:
            latest_time = df["timestamp"].max()
            st.metric("Last Event Time", latest_time)

        # Show recent events
        st.subheader("Most Recent Events")
        st.dataframe(df.sort_values("timestamp", ascending=False).head(20))

        # Risk level distribution
        st.subheader("Risk Level Distribution")
        risk_counts = df["predicted_risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]

        st.bar_chart(
            data=risk_counts.set_index("risk_level")["count"]
        )

        # Optional: filter by area_type or purpose
        st.subheader("Filter by Area Type / Purpose")
        area_types = sorted(df["area_type"].dropna().unique())
        purposes = sorted(df["purpose"].dropna().unique())

        selected_area = st.selectbox("Select area_type (optional)", ["All"] + area_types)
        selected_purpose = st.selectbox("Select purpose (optional)", ["All"] + purposes)

        df_filtered = df.copy()

        if selected_area != "All":
            df_filtered = df_filtered[df_filtered["area_type"] == selected_area]

        if selected_purpose != "All":
            df_filtered = df_filtered[df_filtered["purpose"] == selected_purpose]

        st.write(f"Filtered events: {len(df_filtered)}")
        st.dataframe(df_filtered.sort_values("timestamp", ascending=False).head(20))