import streamlit as st
import pandas as pd
import plotly.express as px

from ai_core.data_pipeline import preprocess_data
from ai_core.model_training import (
    train_random_forest,
    train_lead_time_model
)
from agentic_ai.forecast_agent import ForecastAgent
from agentic_ai.reasoning_agent import ReasoningAgent
from ai_core.explainability import (
    compute_shap_values, plot_global_importance)
import shap
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Health AI | Forecast Agent",
    layout="wide",
    page_icon="📦"
)

st.title("📦 Health Supply Chain Forecast Agent")
st.caption("Enterprise-grade demand & lead-time intelligence")

st.markdown("---")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("⚙️ Configuration")

forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    [7, 14, 30, 60]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Supply Chain CSV",
    type=["csv"]
)

run_button = st.sidebar.button("🚀 Run Forecast Agent")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df_raw.head())

    # FILTERS
    st.sidebar.subheader("🔎 Filters")

    facility = st.sidebar.selectbox(
        "Facility",
        ["All"] + sorted(df_raw["facility"].unique())
    )

    item = st.sidebar.selectbox(
        "Item",
        ["All"] + sorted(df_raw["item"].unique())
    )

    if facility != "All":
        df_raw = df_raw[df_raw["facility"] == facility]

    if item != "All":
        df_raw = df_raw[df_raw["item"] == item]

# --------------------------------------------------
# PIPELINE
# --------------------------------------------------
if uploaded_file and run_button:

    st.markdown("---")
    st.header("📊 Forecast Results")

    df = preprocess_data(
        df_raw,
        date_col="date",
        target_col="demand"
    )

    feature_cols = [
        "y",
        "stock_on_hand",
        "day_of_week",
        "month"
    ]

    # -------------------------------
    # DEMAND MODEL
    # -------------------------------
    with st.spinner("Training demand model..."):
        demand_model, demand_metrics = train_random_forest(
            df=df,
            feature_cols=feature_cols,
            target_col="y"
        )

    col1, col2 = st.columns(2)
    col1.metric("Demand MAE", f"{demand_metrics['MAE']:.2f}")
    col2.metric("Demand RMSE", f"{demand_metrics['RMSE']:.2f}")


    # -------------------------------
    # LEAD TIME MODEL
    # -------------------------------
    with st.spinner("Training lead-time model..."):
        lead_model, lead_metrics = train_lead_time_model(df)

    col3, col4 = st.columns(2)
    col3.metric("Lead Time MAE", f"{lead_metrics['MAE']:.2f}")
    col4.metric("Lead Time RMSE", f"{lead_metrics['RMSE']:.2f}")

    # -------------------------------
    # FUTURE FORECAST
    # -------------------------------
    agent = ForecastAgent()
    future_df = agent.run_future_forecast(
        model=demand_model,
        df_processed=df,
        periods=forecast_horizon
    )

    st.subheader("📈 Demand Forecast")

    fig = px.line(
        future_df,
        x="ds",
        y="forecast",
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------
    # SHAP EXPLAINABILITY
    # ------------------------------

    st.subheader("🔍 AI Explainability (SHAP)")

    # Compute SHAP values for demand model
    shap_explainer, shap_values, X_sample = agent.compute_shap(demand_model, df, feature_cols)

    # Plot global importance
    st.markdown("**Global Feature Importance (Demand Model)**")
    fig = plot_global_importance(shap_values, X_sample, title="Demand Drivers")
    st.pyplot(fig)

    # -------------------------------
    # LEAD TIME VISUALIZATION
    # -------------------------------
    st.subheader("⏳ Lead Time Distribution")

    fig2 = px.histogram(
        df,
        x="lead_time_days",
        nbins=10
    )
    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------
    # AGENT REASONING
    # -------------------------------
    st.subheader("🧠 Agent Explanation")

    reasoner = ReasoningAgent()
    explanations = reasoner.explain_demand_change(df, future_df)

    for e in explanations:
        st.info(e)

    # --------------------------------------------------
    # MULTI-FACILITY / MULTI-ITEM FORECAST
    # --------------------------------------------------
    st.markdown("---")
    st.subheader("🏭 Multi-Facility & Item Forecast")

    batch_forecast_df = agent.run_batch_forecast(
        df=df,
        model=demand_model,
        periods=forecast_horizon
    )

    st.dataframe(batch_forecast_df.tail(20), use_container_width=True)

    # --------------------------------------------------
    # REORDER POINT & SAFETY STOCK
    # --------------------------------------------------
    from agentic_ai.reorder_agent import ReorderAgent

    st.subheader("📦 Reorder Points & Safety Stock")

    reorder_agent = ReorderAgent()
    reorder_df = reorder_agent.compute_reorder_point(batch_forecast_df)

    st.dataframe(reorder_df, use_container_width=True)

    # --------------------------------------------------
    # EXECUTIVE NARRATIVE
    # --------------------------------------------------
    from agentic_ai.narrative_agent import NarrativeAgent

    st.subheader("📝 Executive Summary for COO")

    narrative_agent = NarrativeAgent()
    summary = narrative_agent.generate_summary(reorder_df)

    st.markdown(summary)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("© Health AI Ecosystem | Agentic Supply Chain Intelligence")


