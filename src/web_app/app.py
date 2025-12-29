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


    # Sidebar filters
    st.sidebar.subheader("🔎 Filters")

    facility = "All"
    item = "All"

    if not batch_mode and df_raw is not None:

        if "facility" in df_raw.columns:
            facility = st.sidebar.selectbox(
                "Facility",
                ["All"] + safe_unique_sorted(df_raw["facility"]),
                key="facility_filter"
            )

        if "item" in df_raw.columns:
            item = st.sidebar.selectbox(
                "Item",
                ["All"] + safe_unique_sorted(df_raw["item"]),
                key="item_filter"
            )

    else:
        st.sidebar.caption("Filters disabled in batch mode")

    if df_raw is not None and run_button:

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

        if df_filtered.empty:
            st.warning("No data after applying filters.")
            st.stop()


# =============================
# PIPELINE (API DRIVEN)
# =============================
if df_filtered is not None and run_button:

    st.markdown("---")
    st.header("📊 Results (Computed by FastAPI)")
    st.caption(
        f"Mode: {'Batch' if batch_mode else 'Filtered'} | "
        f"Facility: {facility} | Item: {item}"
    )


    # Optional inventory column
    stock_col = pick_col(
        df_filtered,
        ["stock_on_hand", "current_stock", "on_hand", "stock"]
    )

    big_run = len(df_filtered) > 8000  # you can tune this

    # -------------------------
    # FORECAST + REORDER API
    # -------------------------
    payload = {
        "data": df_filtered.to_dict(orient="records"),
        "date_col": "date",
        "demand_col": "demand",
        "horizon": int(forecast_horizon),
        "stock_col": stock_col,
        
        # ✅ summary mode when big
        "return_forecast_detail": False if big_run else True,
        "return_inventory_detail": False if big_run else True,
        "max_detail_rows": 20000
    }

    try:
        with st.spinner("🚀 Running batch forecast & reorder logic..."):
            result = call_api("/forecast/batch", payload)

        forecast_df = pd.DataFrame(result.get("forecast", []))
        reorder_df = pd.DataFrame(result.get("reorder", []))
        performance_df = pd.DataFrame(result.get("performance", []))
        inventory_df = pd.DataFrame(result.get("inventory", []))
        confidence_df = pd.DataFrame(result.get("confidence", []))
        explanations = result.get("reorder_explanations", [])
        driver_scores_df = pd.DataFrame(result.get("reorder_driver_scores", []))
        scenario_df = pd.DataFrame(result.get("scenarios", []))
        inventory_worst_df = pd.DataFrame(result.get("inventory_worst", []))
        risk_rollup_df = pd.DataFrame(result.get("risk_rollup", []))
        
        
    except Exception as e:
        st.error(f"❌ API call failed: {e}")
        st.stop()

    # =============================
    st.markdown("---")
    st.subheader("📉 Inventory Simulation (Projected Stock + Reorder Timing)")

    if inventory_df.empty:
        st.info("Inventory daily detail omitted (summary mode). Use filtered mode or lower data size to view full daily simulation.")
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
        
        # =============================
        # 🚨 IMMINENT STOCKOUT (WORST-CASE OVER HORIZON)
        # =============================
        worst = (
            inventory_df
            .groupby(["facility", "item"], as_index=False)
            .agg(
                min_stock=("stock_on_hand", "min"),
                min_days_cover=("days_of_cover", "min"),
                any_reorder=("reorder_now", "max"),
            )
        )

        imminent = worst[
            (worst["min_stock"] <= 0) | (worst["min_days_cover"] <= 3)
        ]

        st.metric(
            "🚨 Items at Risk of Stockout (≤3 days)",
            len(imminent)
        )

        st.dataframe(
            imminent.sort_values("min_days_cover"),
            use_container_width=True
        )
        # =============================
        # 🔴 ZERO-STOCK HIGHLIGHT
        # =============================
        def highlight_zero_stock(row):
            return [
                "background-color: #ffcccc"
                if row.get("stock_on_hand", 1) <= 0
                else ""
            ] * len(row)

        st.markdown("### 🔴 Zero Stock Highlight")
        st.dataframe(
            worst.style.apply(highlight_zero_stock, axis=1),
            use_container_width=True
        )


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

            # =============================
            # 🔑 DERIVE ITEM-LEVEL INVENTORY RISK (WORST CASE)
            # =============================
            risk_rollup = (
                inventory_df
                .copy()
                .assign(
                    risk_level=lambda d: d["inventory_risk"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .map({"LOW": 1, "MEDIUM": 2, "HIGH": 3})
                )
                .groupby(["facility", "item"], as_index=False)
                .agg(
                    max_risk=("risk_level", "max"),
                    min_days_cover=("days_of_cover", "min"),
                    any_reorder=("reorder_now", "max"),
                )
            )

            risk_rollup["inventory_risk"] = (
                risk_rollup["max_risk"]
                .map({1: "LOW", 2: "MEDIUM", 3: "HIGH"})
                .fillna("UNKNOWN")
            )
    

            risk_counts = risk_rollup["inventory_risk"].value_counts()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🔴 High Risk", int(risk_counts.get("HIGH", 0)))
            col2.metric("🟠 Medium Risk", int(risk_counts.get("MEDIUM", 0)))
            col3.metric("🟢 Low Risk", int(risk_counts.get("LOW", 0)))
            col4.metric("⚪ Unknown", int(risk_counts.get("UNKNOWN", 0)))   
        else:
            st.error("Inventory risk not available in inventory_df.")
    
        # Show urgent reorder flags
        urgent = inventory_df[inventory_df["reorder_now"] == True].copy()
        st.markdown("**Reorder triggers (rows where reorder_now=True)**")
        st.dataframe(urgent.tail(200), use_container_width=True)

        # Summary table: lowest days of cover
        # =============================
        # 🚨 MOST EXPOSED ITEMS (LOWEST DAYS OF COVER)
        # =============================
        st.markdown("### 🚨 Most Exposed Items (Lowest Days of Cover)")

        exposed = (
            risk_rollup
            .sort_values("min_days_cover")
            .head(20)
        )
        st.dataframe(
        exposed[
                ["facility", "item", "min_days_cover", "inventory_risk", "any_reorder"]
            ],
            use_container_width=True
        )
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

    # --------------------------------------------------
    # Merge confidence into reorder recommendations
    # --------------------------------------------------
    if not reorder_df.empty and not confidence_df.empty:
        reorder_df = reorder_df.merge(
            confidence_df[
                ["facility", "item", "confidence_score", "confidence_band"]
            ],
            on=["facility", "item"],
            how="left"
        )
    
    # =============================
    # REORDER DISPLAY
    # =============================
    st.markdown("---")
    st.subheader("📦 Reorder Points & Safety Stock")

    if reorder_df.empty:
        st.warning("No reorder recommendations returned.")
    else:
        display_df = reorder_df.copy()
        if "confidence_band" in display_df.columns:
            display_df["confidence"] = (
                display_df["confidence_band"]
                + " ("
                + display_df["confidence_score"].astype(str)
                + ")"
            )

        if "reorder_point" in reorder_df.columns:
            top = reorder_df.sort_values(
                "reorder_point",
                ascending=False
            ).head(10)

            st.markdown("**Top 10 highest reorder risks**")
            st.dataframe(top, use_container_width=True)


        # =============================
        # EXPLAINABLE REORDER PANEL
        # =============================
        st.markdown("---")
        st.subheader("🔍 Why these Reorder Points? (Explainability)")

        if not explanations:
            st.info("No explanations available.")
        else:
            # Show explanations as bullet points
            for exp in explanations[:15]:
                st.markdown(f"- {exp}")
                
            if not driver_scores_df.empty:
                st.markdown("### 📊 Reorder Drivers Breakdown")

                fig = px.bar(
                    driver_scores_df,
                    x=["demand_score", "lead_time_score", "variability_score"],
                    y="item",
                    orientation="h",
                    title="Relative Contribution to Reorder Point",
                )
                st.plotly_chart(fig, use_container_width=True)

    
    # =============================
    # SCENARIO ANALYSIS
    # =============================
    st.markdown("---")
    st.subheader("🧪 What-if Scenarios")

    if scenario_df.empty:
        st.info("No scenario results available.")
    else:
        scenario_items = sorted(scenario_df["item"].unique().tolist())
        scenario_facilities = sorted(scenario_df["facility"].unique().tolist())
        
        # 🔑 Use dynamic keys to force refresh in batch mode
        item_key = "scenario_item_batch" if batch_mode else "scenario_item_filtered"
        facility_key = "scenario_facility_batch" if batch_mode else "scenario_facility_filtered"
        
        selected_item = st.selectbox(
            "Select item",
            scenario_items,
            key=item_key
        )
        selected_facility = st.selectbox(
            "Select facility",
            scenario_facilities,
            key=facility_key
        )

        
        # 🔄 Auto-sync ONLY when NOT in batch mode
        if not batch_mode:
            if facility != "All" and facility in scenario_facilities:
                selected_facility = facility

            if item != "All" and item in scenario_items:
                selected_item = item
            
        view = scenario_df[
            (scenario_df["item"] == selected_item) &
            (scenario_df["facility"] == selected_facility)
        ]

        st.dataframe(
            view[
                ["scenario", "avg_daily_demand", "lead_time_days",
                "safety_stock", "reorder_point"]
            ],
            use_container_width=True
        )

        fig = px.bar(
            view,
            x="scenario",
            y="reorder_point",
            title="Reorder Point by Scenario"
        )
        st.plotly_chart(fig, use_container_width=True)
    
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
                "inventory_risk": risk_rollup.to_dict(orient="records"),
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



