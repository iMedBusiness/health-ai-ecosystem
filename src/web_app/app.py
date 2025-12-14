import streamlit as st
import pandas as pd
import plotly.express as px

from ai_core.data_pipeline import load_data, preprocess_data
from ai_core.model_training import train_random_forest
from agentic_ai.forecast_agent import ForecastAgent

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Ashraf AI Ecosystem | Forecast Agent",
    layout="wide",
    page_icon="📦"
)

st.title("📦 Health Supply Chain Forecasting")
st.caption("Powered by Ashraf AI Ecosystem")

st.markdown("---")

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("⚙️ Controls")

forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon (days)",
    [7, 14, 30, 60]
)

run_button = st.sidebar.button("🚀 Train & Forecast")

# --------------------------------------------------
# DATA UPLOAD
# --------------------------------------------------
st.header("1️⃣ Upload Supply Chain Data")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.success(f"Loaded {df_raw.shape[0]} rows")

    st.subheader("Preview")
    st.dataframe(df_raw.head())

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
if uploaded_file and run_button:

    st.markdown("---")
    st.header("2️⃣ Model Training & Forecasting")

    with st.spinner("Preprocessing data..."):
        df = preprocess_data(
            df_raw,
            date_col="date",
            target_col="demand"
        )

    st.success("Data preprocessed")

    feature_cols = [
        "y",
        "stock_on_hand",
        "day_of_week",
        "month"
    ]

    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------
    with st.spinner("Training Random Forest model..."):
        model, metrics = train_random_forest(
            df=df,
            feature_cols=feature_cols,
            target_col="y"
        )

    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{metrics['MAE']:.2f}")
    col2.metric("RMSE", f"{metrics['RMSE']:.2f}")

    # --------------------------------------------------
    # FUTURE FORECAST
    # --------------------------------------------------
    agent = ForecastAgent()

    future_df = agent.run_future_forecast(
        model=model,
        df_processed=df,
        periods=forecast_horizon
    )

    st.markdown("---")
    st.header("3️⃣ Future Demand Forecast")

    fig = px.line(
        future_df,
        x="ds",
        y="forecast",
        title=f"Next {forecast_horizon} Days Demand Forecast",
        labels={"forecast": "Predicted Demand", "ds": "Date"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Table")
    st.dataframe(future_df[["ds", "forecast"]])

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("© Ashraf AI Ecosystem | Agentic Supply Chain Intelligence")

