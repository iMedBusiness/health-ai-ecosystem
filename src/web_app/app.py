import streamlit as st
import pandas as pd
import plotly.express as px
import requests

# -----------------------------
# CONFIG
# -----------------------------
FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Health AI | Forecast Agent",
    layout="wide",
    page_icon="📦"
)

# -----------------------------
# HELPERS
# -----------------------------
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


# -----------------------------
# HEADER
# -----------------------------
st.title("📦 Health Supply Chain Forecast Agent")
st.caption("Enterprise-grade demand, lead-time, reorder & risk intelligence (API-powered)")
st.markdown("---")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Configuration")
forecast_horizon = st.sidebar.selectbox("Forecast Horizon (days)", [7, 14, 30, 60])
batch_mode = st.sidebar.checkbox("📦 Batch mode (all facilities & items)", value=True)

uploaded_file = st.sidebar.file_uploader("Upload Supply Chain CSV", type=["csv"])
run_button = st.sidebar.button("🚀 Run (via FastAPI)")

st.sidebar.markdown("---")
st.sidebar.caption("Backend must be running at:")
st.sidebar.code(FASTAPI_URL)

# -----------------------------
# LOAD DATA
# -----------------------------
df_raw = None
if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df_raw.head(20), use_container_width=True)

    # ---- Required fields check (soft) ----
    # We'll still try to run but show warnings.
    required = ["facility", "item", "date", "demand"]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        st.warning(f"Missing expected columns: {missing}. "
                   "FastAPI endpoint expects at least facility, item, date, demand.")

    # ---- Filters (only when NOT batch mode) ----
    st.sidebar.subheader("🔎 Filters")
    if "facility" in df_raw.columns:
        facility = st.sidebar.selectbox("Facility", ["All"] + safe_unique_sorted(df_raw["facility"]))
    else:
        facility = "All"

    if "item" in df_raw.columns:
        item = st.sidebar.selectbox("Item", ["All"] + safe_unique_sorted(df_raw["item"]))
    else:
        item = "All"

    df_filtered = df_raw.copy()
    if not batch_mode:
        if facility != "All" and "facility" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["facility"].astype(str).str.strip() == facility]
        if item != "All" and "item" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["item"].astype(str).str.strip() == item]
else:
    df_filtered = None

# -----------------------------
# PIPELINE (API-DRIVEN)
# -----------------------------
if df_filtered is not None and run_button:
    st.markdown("---")
    st.header("📊 Results (Computed by FastAPI)")

    # ---- Inventory column detection for simulation (optional) ----
    stock_col = pick_col(df_filtered, ["stock_on_hand", "current_stock", "on_hand", "stock"])

    payload = {
        "data": df_filtered.to_dict(orient="records"),
        "date_col": "date",
        "demand_col": "demand",
        "horizon": int(forecast_horizon),
        # optional: tell backend which stock column exists
        "stock_col": stock_col
    }

    # ---- Call backend forecast+reorder (Milestone 2A core) ----
    try:
        with st.spinner("🚀 Calling FastAPI: /forecast/batch ..."):
            result = call_api("/forecast/batch", payload)

        forecast_df = pd.DataFrame(result.get("forecast", []))
        reorder_df = pd.DataFrame(result.get("reorder", []))

        # -----------------------------
        # ⚡ FORECAST ENGINE PERFORMANCE
        # -----------------------------
        if "performance" in result:
            st.markdown("---")
            st.subheader("⚡ Forecast Engine Performance")

            perf_df = pd.DataFrame(result["performance"])

            if not perf_df.empty:
                col1, col2 = st.columns(2)

                if "cache_hit" in perf_df.columns:
                    col1.metric(
                        "Cache Hit Rate",
                        f"{perf_df['cache_hit'].mean():.0%}"
                    )
                else:
                    col1.metric("Cache Hit Rate", "N/A")
                if "runtime_sec" in perf_df.columns:
                    col2.metric(
                        "Avg Runtime (sec)",
                        f"{perf_df['runtime_sec'].mean():.2f}"
                    )
                else:
                    col2.metric("Avg Runtime (sec)", "N/A")
                st.dataframe(perf_df, use_container_width=True)
            else:
                st.info("No performance metrics returned by API.")


        # -----------------------------
        # FORECAST DISPLAY
        # -----------------------------
        st.subheader("📈 Demand Forecast (Batch)")
        if forecast_df.empty:
            st.warning("No forecast rows returned by API.")
        else:
            # Make sure ds exists for plotting
            if "ds" in forecast_df.columns:
                forecast_df["ds"] = pd.to_datetime(forecast_df["ds"], errors="coerce")
                forecast_df = forecast_df.dropna(subset=["ds"]).sort_values("ds")

            # Simple chart: per item (and facility if many)
            color_col = "item" if "item" in forecast_df.columns else None
            facet_col = "facility" if ("facility" in forecast_df.columns and forecast_df["facility"].nunique() <= 6) else None

            if facet_col:
                fig = px.line(
                    forecast_df,
                    x="ds",
                    y="forecast",
                    color=color_col,
                    facet_row=facet_col,
                    markers=False,
                    title="Forecast over time (faceted by facility)"
                )
            else:
                fig = px.line(
                    forecast_df,
                    x="ds",
                    y="forecast",
                    color=color_col,
                    markers=False,
                    title="Forecast over time"
                )

            fig.update_traces(mode="lines")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Forecast table (last 100 rows)**")
            st.dataframe(forecast_df.tail(100), use_container_width=True)


        # -----------------------------
        # REORDER DISPLAY
        # -----------------------------
        st.subheader("📦 Reorder Points & Safety Stock")
        if reorder_df.empty:
            st.warning("No reorder rows returned by API.")
        else:
            st.dataframe(reorder_df, use_container_width=True)

            # Highlight top risk
            if "reorder_point" in reorder_df.columns:
                top = reorder_df.sort_values("reorder_point", ascending=False).head(10)
                st.markdown("**Top 10 highest reorder points**")
                st.dataframe(top, use_container_width=True)

    except Exception as e:
        st.error(f"❌ API call failed: {e}")
        st.stop()

    # -----------------------------
    # OPTIONAL: Inventory simulation + COO narrative via API
    # (These require you to add endpoints. We show UI now, and wire later.)
    # -----------------------------
    st.markdown("---")
    st.subheader("📉 Inventory Simulation (optional, via API)")
    st.caption("Next step: add /simulate/inventory endpoint and enable this section.")

    st.markdown("---")
    st.subheader("🧠 Executive Summary (COO View)")

    try:
        payload = {
            "reorder": reorder_df.to_dict(orient="records"),
            "horizon_days": forecast_horizon
        }

        with st.spinner("🧠 Generating executive narrative..."):
            exec_result = call_api(
                "/executive/summary",
                payload
            )

        st.markdown(exec_result["summary"])

    except Exception as e:
        st.warning(f"Executive narrative unavailable: {e}")


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("© Health AI Ecosystem | Agentic Supply Chain Intelligence (Streamlit client + FastAPI backend)")



