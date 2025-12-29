# apps/backend/api/forecast.py

from fastapi import APIRouter, HTTPException
import pandas as pd

from apps.backend.schemas.requests import BatchForecastRequest
from src.agentic_ai.forecast_agent import ForecastAgent
from src.agentic_ai.reorder_agent import ReorderAgent
from src.ai_core.data_pipeline import preprocess_data
from src.ai_core.volatility import classify_volatility
from src.agentic_ai.inventory_simulation_agent import InventorySimulationAgent
from src.agentic_ai.inventory_risk_agent import InventoryRiskAgent
from src.agentic_ai.data_quality_agent import DataQualityAgent
from src.agentic_ai.confidence_agent import ConfidenceAgent
from src.agentic_ai.explainable_reorder import ExplainableReorderAgent
from src.agentic_ai.scenario_agent import ScenarioAgent


router = APIRouter()


def _normalize_keys(df: pd.DataFrame, cols=("facility", "item")) -> pd.DataFrame:
    """Normalize merge keys consistently across all tables."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    return df


@router.post("/batch")
def batch_forecast(request: BatchForecastRequest):
    """
    Multi-item, multi-facility batch forecast endpoint.
    Fully self-contained: validates, preprocesses, forecasts, reorders.
    """
    # --------------------------------------------------
    # 1) Load payload
    # --------------------------------------------------
    df_raw = pd.DataFrame(request.data)
    if df_raw.empty:
        raise HTTPException(status_code=400, detail="Empty dataset received")

    # --------------------------------------------------
    # 2) Validate required cols
    # --------------------------------------------------
    required_cols = {"facility", "item", request.date_col, request.demand_col}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Missing required columns",
                "missing": sorted(list(missing)),
                "received": df_raw.columns.tolist(),
            },
        )

    # ✅ Normalize raw keys ASAP (so everything downstream is consistent)
    df_raw = _normalize_keys(df_raw, ("facility", "item"))

    # Make lead_time numeric if present
    if "lead_time_days" in df_raw.columns:
        df_raw["lead_time_days"] = pd.to_numeric(df_raw["lead_time_days"], errors="coerce")

    # --------------------------------------------------
    # 3) Preprocess (creates ds, y, day_of_week, month)
    # --------------------------------------------------
    df = preprocess_data(
        df_raw,
        date_col=request.date_col,
        target_col=request.demand_col,
    )

    # Ensure preprocessed keys still normalized (defensive)
    df = _normalize_keys(df, ("facility", "item"))

    # --------------------------------------------------
    # 4) Volatility classification (FIXED & ALIGNED)
    # --------------------------------------------------
    try:
        vol_df = classify_volatility(df, y_col="y", group_cols=("facility", "item"))

        # 🔑 Map numeric-style volatility to planning semantics
        vol_map = {
            "Low": "Stable",
            "Medium": "Seasonal",
            "High": "Erratic",
        }

        vol_df["volatility_class"] = vol_df["volatility"].map(vol_map)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Volatility classification failed: {str(e)}"
        )

    # --------------------------------------------------
    # 5) Agents
    # --------------------------------------------------
    forecast_agent = ForecastAgent()
    reorder_agent = ReorderAgent()

    # --------------------------------------------------
    # 6) Batch forecast
    # --------------------------------------------------
    try:
        output = forecast_agent.run_batch_forecast(
            df=df,
            periods=request.horizon,
            parallel=True,
            max_workers=4
        )
        batch_forecast_df = output["forecast"]
        metrics_df = output["metrics"]

        # ✅ Normalize forecast keys too (in case ForecastAgent changes formatting)
        batch_forecast_df = _normalize_keys(batch_forecast_df, ("facility", "item"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch forecasting failed: {str(e)}")

    # --------------------------------------------------
    # 7) Lead-time DF (from RAW) + normalize BEFORE merge
    # --------------------------------------------------
    if "lead_time_days" in df_raw.columns:
        lead_time_df = (
            df_raw
            .groupby(["facility", "item"], as_index=False)
            .agg({"lead_time_days": "mean"})
        )
    else:
        lead_time_df = pd.DataFrame(columns=["facility", "item", "lead_time_days"])

    lead_time_df = _normalize_keys(lead_time_df, ("facility", "item"))

    # --------------------------------------------------
    # 8) Merge lead-time into forecast (FIXED)
    # --------------------------------------------------
    if not lead_time_df.empty:
        batch_forecast_df = batch_forecast_df.merge(
            lead_time_df,
            on=["facility", "item"],
            how="left"
        )

    # Guardrails
    if "lead_time_days" not in batch_forecast_df.columns:
        raise HTTPException(
            status_code=500,
            detail="lead_time_days missing after merge — cannot compute reorder points"
        )

    missing_lt = int(batch_forecast_df["lead_time_days"].isna().sum())
    # ✅ Instead of failing all, we allow partial + compute reorder where possible
    # (much better UX)
    # If you want strict mode, change this to raise.
    # For now: keep going.

    # --------------------------------------------------
    # 9) Reorder points
    # --------------------------------------------------
    try:
        reorder_df = reorder_agent.compute_reorder_point(
            forecast_df=batch_forecast_df,
            demand_col="forecast",
            lead_time_col="lead_time_days",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reorder computation failed: {str(e)}")

    # --------------------------------------------------
    # 9.1) Explainable reorder drivers
    # --------------------------------------------------
    explain_agent = ExplainableReorderAgent()

    reorder_explanations = explain_agent.explain_reorder_drivers(reorder_df)
    reorder_driver_scores = explain_agent.compute_driver_scores(reorder_df)

    # --------------------------------------------------
    # 9.2) Scenario analysis
    # --------------------------------------------------
    scenario_agent = ScenarioAgent()
    scenario_results = []

    # --- Baseline
    baseline = reorder_df.copy()
    baseline["scenario"] = "Baseline"
    scenario_results.append(baseline)

    # --- Demand surge (+30%)
    surge_forecast = scenario_agent.run_demand_surge(
        batch_forecast_df,
        surge_pct=0.30
    )
    surge_reorder = reorder_agent.compute_reorder_point(
        forecast_df=surge_forecast,
        demand_col="forecast",
        lead_time_col="lead_time_days",
    )
    surge_reorder["scenario"] = "Demand +30%"
    scenario_results.append(surge_reorder)

    # --- Lead time shock (+7 days)
    lt_forecast = scenario_agent.run_lead_time_shock(
        batch_forecast_df,
        extra_days=7
    )
    lt_reorder = reorder_agent.compute_reorder_point(
        forecast_df=lt_forecast,
        demand_col="forecast",
        lead_time_col="lead_time_days",
    )
    lt_reorder["scenario"] = "Lead Time +7d"
    scenario_results.append(lt_reorder)

    scenario_df = pd.concat(scenario_results, ignore_index=True)

    # --------------------------------------------------
    # 10) Build inventory_df (starting stock) for simulation
    # --------------------------------------------------

    inventory_df = pd.DataFrame(columns=["facility", "item", "stock_on_hand"])

    # 1️⃣ Explicit stock column (preferred)
    if request.stock_col and request.stock_col in df_raw.columns:
        stock_col = request.stock_col

    # 2️⃣ Fallback auto-detection
    else:
        possible_stock_cols = [
            "stock_on_hand",
            "current_stock",
            "on_hand",
            "stock"
        ]
        stock_col = next(
            (c for c in possible_stock_cols if c in df_raw.columns),
            None
        )

    if stock_col:
        inventory_df = (
            df_raw
            .sort_values(request.date_col)
            .groupby(["facility", "item"], as_index=False)
            .agg({stock_col: "last"})
            .rename(columns={stock_col: "stock_on_hand"})
        )
    if inventory_df.empty:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Inventory simulation skipped",
                "reason": "No usable stock column detected",
                "received_columns": df_raw.columns.tolist(),
                "expected_columns": possible_stock_cols,
                "request_stock_col": request.stock_col
            }
        )
    # --------------------------------------------------
    # 11) Inventory simulation
    # --------------------------------------------------
    sim_df = pd.DataFrame()
    if "stock_on_hand" not in inventory_df.columns:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid inventory dataframe",
                "reason": "stock_on_hand column missing",
                "inventory_columns": inventory_df.columns.tolist()
            }
        )
    if not batch_forecast_df.empty and not reorder_df.empty and not inventory_df.empty:
        sim_agent = InventorySimulationAgent()
        sim_df = sim_agent.simulate(
            forecast_df=batch_forecast_df,
            inventory_df=inventory_df,
            reorder_df=reorder_df,
            stock_col="stock_on_hand"
        )
    
    # --------------------------------------------------
    # 11.1) Merge volatility into inventory simulation
    # --------------------------------------------------
    if not sim_df.empty and not vol_df.empty:
        for c in ["facility", "item"]:
            sim_df[c] = sim_df[c].astype(str).str.strip().str.lower()
            vol_df[c] = vol_df[c].astype(str).str.strip().str.lower()

        sim_df = sim_df.merge(
            vol_df[["facility", "item", "volatility_class"]],
            on=["facility", "item"],
            how="left"
        )
    if not sim_df.empty:
        risk_agent = InventoryRiskAgent()
        sim_df = risk_agent.score(sim_df)    
        
    if sim_df["volatility_class"].isna().all():
        raise RuntimeError(
            "Volatility missing for all rows — risk scoring would be invalid"
        )
    
   
    # --------------------------------------------------
    # 11.2) Confidence scoring & data quality guardrails
    # --------------------------------------------------
    confidence_results = []

    dq_agent = DataQualityAgent()
    conf_agent = ConfidenceAgent()

    # Group on PREPROCESSED df (has ds, y)
    for (facility, item), g in df.groupby(["facility", "item"]):
        # -----------------------------
        # Data Quality
        # -----------------------------
        dq = dq_agent.assess(g)

        # -----------------------------
        # Volatility class
        # -----------------------------
        vol_match = vol_df.loc[
            (vol_df["facility"] == facility) &
            (vol_df["item"] == item),
            "volatility_class"
        ].values
        volatility_class = vol_match[0] if len(vol_match) else "Unknown"

        # -----------------------------
        # Forecast performance (cache hit)
        # -----------------------------
        perf = metrics_df.loc[
            (metrics_df["facility"] == facility) &
            (metrics_df["item"] == item)
        ]
        forecast_cache_hit = (
            bool(perf["cache_hit"].iloc[0])
            if not perf.empty and "cache_hit" in perf.columns
            else False
        )

        # -----------------------------
        # Lead time confidence
        # -----------------------------
        lt_rows = batch_forecast_df.loc[
            (batch_forecast_df["facility"] == facility) &
            (batch_forecast_df["item"] == item),
            "lead_time_days"
        ]
        lead_time_missing = lt_rows.isna().any()

        # -----------------------------
        # Confidence score
        # -----------------------------
        confidence = conf_agent.score(
            data_quality=dq,
            volatility_class=volatility_class,
            lead_time_missing=lead_time_missing,
            forecast_cache_hit=forecast_cache_hit
        )

        confidence_results.append({
            "facility": facility,
            "item": item,
            **confidence
        })

    # --------------------------------------------------
    # 11.3) Rollups (ALWAYS returned, lightweight)
    # --------------------------------------------------
    inventory_worst = pd.DataFrame()
    risk_rollup = pd.DataFrame()

    if not sim_df.empty and "stock_on_hand" in sim_df.columns:
        inventory_worst = (
            sim_df.groupby(["facility", "item"], as_index=False)
            .agg(
                min_stock=("stock_on_hand", "min"),
                min_days_cover=("days_of_cover", "min"),
                any_reorder=("reorder_now", "max"),
            )
        )

    if not sim_df.empty and "inventory_risk" in sim_df.columns:
        risk_rollup = (
            sim_df.assign(
                risk_level=sim_df["inventory_risk"]
                .astype(str).str.strip().str.upper()
                .map({"LOW": 1, "MEDIUM": 2, "HIGH": 3})
            )
            .groupby(["facility", "item"], as_index=False)
            .agg(max_risk=("risk_level", "max"))
        )
        risk_rollup["inventory_risk"] = (
            risk_rollup["max_risk"]
            .map({1: "LOW", 2: "MEDIUM", 3: "HIGH"})
            .fillna("UNKNOWN")
        )


    # --------------------------------------------------
    # RESPONSE SIZE CONTROL (SUMMARY MODE)
    # --------------------------------------------------

    # Default outputs
    forecast_out = batch_forecast_df
    inventory_out = sim_df
    scenario_out = scenario_df


    # --------------------------------------------------
    # 12) Response
    # --------------------------------------------------
    return {
        "status": "success",
        "meta": {
            "horizon_days": request.horizon,
            "records_received": len(df_raw),
            "forecast_rows": len(forecast_out),
            "inventory_rows": len(inventory_out),
            "scenario_rows": len(scenario_out),
        
            "reorder_rows": len(reorder_df),
            "cache_hit_rate": round(float(metrics_df["cache_hit"].mean()), 2) if "cache_hit" in metrics_df.columns and not metrics_df.empty else None,
            "avg_runtime_sec": round(float(metrics_df["runtime_sec"].mean()), 2) if "runtime_sec" in metrics_df.columns and not metrics_df.empty else None,
            "detail_mode": {
                "forecast": bool(request.return_forecast_detail),
                "inventory": bool(request.return_inventory_detail),
                "max_detail_rows": int(request.max_detail_rows),
            },
        },
        # ✅ use the controlled outputs
        "forecast": forecast_out.to_dict(orient="records"),
        "inventory": inventory_out.to_dict(orient="records"),
        "scenarios": scenario_out.to_dict(orient="records"),

        # Always returned

        "reorder": reorder_df.to_dict(orient="records"),
        "performance": metrics_df.to_dict(orient="records"),
        "volatility": vol_df.to_dict(orient="records"),
        "confidence": confidence_results,
        "reorder_explanations": reorder_explanations,
        "reorder_driver_scores": reorder_driver_scores.to_dict(orient="records"),
    }
