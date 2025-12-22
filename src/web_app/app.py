import streamlit as st
import pandas as pd
import plotly.express as px
import requests

# =============================
# CONFIG
# =============================
FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Health AI | Forecast Agent",
    layout="wide",
    page_icon="📦"
)

# =============================
# HELPERS
# =============================
def call_api(path: str, payload: dict, timeout: int = 180) -> dict:
    url = f"{FASTAPI_URL}{path}"
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"API {path} failed ({r.status_code}): {r.text}")
    return r.json()


def safe_unique_sorted(series: pd.Series):
    return sorted(series.dropna().astype(str).str.strip().unique().tolist())


def pick_col(df: pd.DataFrame, options: list[str]):
    for c in options:
        if c in df.columns:
            return c
    return None


# =============================
# HEADER
# =============================
st.title("📦 Health Supply Chain Forecast Agent")
st.caption(
    "Enterprise-grade demand forecasting, reorder optimization "
    "and executive intelligence (FastAPI-powered)"
)
st.markdown("---")

# =============================
# SIDEBAR
# =============================
st.sidebar.header("⚙️ Configuration")

forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon (days)",
    [7, 14, 30, 60],
    index=2
)

batch_mode = st.sidebar.checkbox(
    "📦 Batch mode (all facilities & items)",
    value=True
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Supply Chain CSV",
    type=["csv"]
)

run_button = st.sidebar.button("🚀 Run via FastAPI")

st.sidebar.markdown("---")
st.sidebar.caption("Backend must be running at:")
st.sidebar.code(FASTAPI_URL)

# =============================
# LOAD DATA
# =============================
df_raw = None
df_filtered = None

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    # Soft validation
    expected = ["facility", "item", "date", "demand"]
    missing = [c for c in expected if c not in df_raw.columns]
    if missing:
        st.warning(
            f"Missing expected columns: {missing}. "
            "Forecasting may still work if backend mapping is correct."
        )

    # Sidebar filters (disabled in batch mode)
    st.sidebar.subheader("🔎 Filters")

    facility = "All"
    item = "All"

    if "facility" in df_raw.columns:
        facility = st.sidebar.selectbox(
            "Facility",
            ["All"] + safe_unique_sorted(df_raw["facility"])
        )

    if "item" in df_raw.columns:
        item = st.sidebar.selectbox(
            "Item",
            ["All"] + safe_unique_sorted(df_raw["item"])
        )

    df_filtered = df_raw.copy()

    if not batch_mode:
        if facility != "All":
            df_filtered = df_filtered[
                df_filtered["facility"].astype(str).str.strip() == facility
            ]
        if item != "All":
            df_filtered = df_filtered[
                df_filtered["item"].astype(str).str.strip() == item
            ]

# =============================
# PIPELINE (API DRIVEN)
# =============================
if df_filtered is not None and run_button:

    st.markdown("---")
    st.header("📊 Results (Computed by FastAPI)")

    # Optional inventory column
    stock_col = pick_col(
        df_filtered,
        ["stock_on_hand", "current_stock", "on_hand", "stock"]
    )

    # -------------------------
    # FORECAST + REORDER API
    # -------------------------
    payload = {
        "data": df_filtered.to_dict(orient="records"),
        "date_col": "date",
        "demand_col": "demand",
        "horizon": int(forecast_horizon),
        "stock_col": stock_col
    }

    try:
        with st.spinner("🚀 Running batch forecast & reorder logic..."):
            result = call_api("/forecast/batch", payload)

        forecast_df = pd.DataFrame(result.get("forecast", []))
        reorder_df = pd.DataFrame(result.get("reorder", []))
        performance_df = pd.DataFrame(result.get("performance", []))
        inventory_df = pd.DataFrame(result.get("inventory", []))
        
        
    except Exception as e:
        st.error(f"❌ API call failed: {e}")
        st.stop()

    st.markdown("---")
    st.subheader("📉 Inventory Simulation (Projected Stock + Reorder Timing)")

    if inventory_df.empty:
        st.warning(
            "Inventory simulation not available.\n\n"
            "Possible reasons:\n"
            "- No valid stock column detected\n"
            "- Stock column contains only null/non-numeric values\n"
            "- Stock not provided per facility–item\n\n"
            "Tip: Ensure stock is provided per row or explicitly select the stock column."
        )
    else:
        inventory_df["ds"] = pd.to_datetime(inventory_df["ds"], errors="coerce")
        inventory_df = inventory_df.dropna(subset=["ds"])

        # Plot on_hand over time
        stock_plot_col = (
            "stock_on_hand"
            if "stock_on_hand" in inventory_df.columns
            else "on_hand" if "on_hand" in inventory_df.columns
            else None
        )

        if stock_plot_col is None:
            st.warning("No stock column available for plotting.")
        else:
            fig_inv = px.line(
                inventory_df,
                x="ds",
                y=stock_plot_col,
                color="item" if "item" in inventory_df.columns else None,
                facet_row=(
                    "facility"
                    if "facility" in inventory_df.columns
                    and inventory_df["facility"].nunique() <= 6
                    else None
                ),
                title="Projected On-hand Inventory"
            )
            st.plotly_chart(fig_inv, use_container_width=True)

        if "inventory_risk" in inventory_df.columns:
            st.markdown("### 🚦 Inventory Risk Levels")
            st.dataframe(
                inventory_df[
                    ["facility", "item", "ds", "days_of_cover", "reorder_now", "inventory_risk"]
                ].sort_values("inventory_risk"),
                use_container_width=True
            )
            
        if "inventory_risk" in inventory_df.columns:
            risk_counts = inventory_df["inventory_risk"].value_counts()

            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 High Risk", risk_counts.get("HIGH", 0))
            col2.metric("🟠 Medium Risk", risk_counts.get("MEDIUM", 0))
            col3.metric("🟢 Low Risk", risk_counts.get("LOW", 0))
    
        # Show urgent reorder flags
        urgent = inventory_df[inventory_df["reorder_now"] == True].copy()
        st.markdown("**Reorder triggers (rows where reorder_now=True)**")
        st.dataframe(urgent.tail(200), use_container_width=True)

        # Summary table: lowest days of cover
        st.markdown("**Top stockout risk (lowest days of cover)**")
        top_risk = inventory_df.sort_values("days_of_cover").groupby(["facility","item"], as_index=False).first().head(20)
        st.dataframe(top_risk, use_container_width=True)

    
    # =============================
    # PERFORMANCE (OPTIONAL)
    # =============================
    if not performance_df.empty:
        st.markdown("---")
        st.subheader("⚡ Forecast Engine Performance")

        col1, col2 = st.columns(2)

        if "cache_hit" in performance_df.columns:
            col1.metric(
                "Cache Hit Rate",
                f"{performance_df['cache_hit'].mean():.0%}"
            )
        else:
            col1.metric("Cache Hit Rate", "N/A")

        if "runtime_sec" in performance_df.columns:
            col2.metric(
                "Avg Runtime (sec)",
                f"{performance_df['runtime_sec'].mean():.2f}"
            )
        else:
            col2.metric("Avg Runtime (sec)", "N/A")

        st.dataframe(performance_df, use_container_width=True)

    # =============================
    # FORECAST VISUALIZATION
    # =============================
    st.markdown("---")
    st.subheader("📈 Demand Forecast")

    if forecast_df.empty:
        st.warning("No forecast rows returned.")
    else:
        forecast_df["ds"] = pd.to_datetime(
            forecast_df["ds"], errors="coerce"
        )
        forecast_df = forecast_df.dropna(subset=["ds"])

        color_col = "item" if "item" in forecast_df.columns else None
        facet_col = (
            "facility"
            if "facility" in forecast_df.columns
            and forecast_df["facility"].nunique() <= 6
            else None
        )

        fig = px.line(
            forecast_df,
            x="ds",
            y="forecast",
            color=color_col,
            facet_row=facet_col,
            title="Forecasted Demand Over Time"
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(forecast_df.tail(100), use_container_width=True)

    # =============================
    # REORDER DISPLAY
    # =============================
    st.markdown("---")
    st.subheader("📦 Reorder Points & Safety Stock")

    if reorder_df.empty:
        st.warning("No reorder recommendations returned.")
    else:
        st.dataframe(reorder_df, use_container_width=True)

        if "reorder_point" in reorder_df.columns:
            top = reorder_df.sort_values(
                "reorder_point",
                ascending=False
            ).head(10)

            st.markdown("**Top 10 highest reorder risks**")
            st.dataframe(top, use_container_width=True)


    # =============================
    # EXECUTIVE SUMMARY (COO)
    # =============================
    st.subheader("🧠 Executive Summary (COO View)")

    if reorder_df is None or reorder_df.empty:
        st.info(
            "Executive summary will appear once reorder points are successfully computed."
        )
    else:
        try:
            payload = {
                "reorder": reorder_df.to_dict(orient="records"),
                "volatility": result.get("volatility", []),
                "horizon_days": int(forecast_horizon)
            }

            with st.spinner("🧠 Generating executive narrative..."):
                exec_result = call_api("/executive/summary", payload)

            st.markdown(exec_result.get("summary", "No summary returned."))

        except Exception as e:
            st.warning(f"Executive narrative unavailable: {e}")


# =============================
# FOOTER
# =============================
st.markdown("---")
st.caption(
    "© Health AI Ecosystem | Agentic Supply Chain Intelligence "
    "(Streamlit Client + FastAPI Backend)"
)



