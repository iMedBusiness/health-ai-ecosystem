# apps/backend/api/forecast.py

from fastapi import APIRouter, HTTPException
import pandas as pd

from apps.backend.schemas.requests import BatchForecastRequest
from src.agentic_ai.forecast_agent import ForecastAgent
from src.agentic_ai.reorder_agent import ReorderAgent
from ai_core.data_pipeline import preprocess_data
from src.ai_core.volatility import classify_volatility
from src.agentic_ai.inventory_simulation_agent import InventorySimulationAgent
from src.agentic_ai.inventory_risk_agent import InventoryRiskAgent

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
    # 4) Volatility classification
    # --------------------------------------------------
    try:
        vol_df = classify_volatility(df, y_col="y", group_cols=("facility", "item"))
        vol_df = vol_df.rename(columns={"volatility": "volatility_class"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Volatility classification failed: {str(e)}")

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
        
    # --------------------------------------------------
    # 11.2) Inventory risk scoring
    # --------------------------------------------------
    if not sim_df.empty:
        risk_agent = InventoryRiskAgent()
        sim_df = risk_agent.score(sim_df)

    # --------------------------------------------------
    # 12) Response
    # --------------------------------------------------
    return {
        "status": "success",
        "meta": {
            "horizon_days": request.horizon,
            "records_received": len(df_raw),
            "forecast_rows": len(batch_forecast_df),
            "reorder_rows": len(reorder_df),
            "cache_hit_rate": round(float(metrics_df["cache_hit"].mean()), 2) if "cache_hit" in metrics_df.columns and not metrics_df.empty else None,
            "avg_runtime_sec": round(float(metrics_df["runtime_sec"].mean()), 2) if "runtime_sec" in metrics_df.columns and not metrics_df.empty else None,
        },
        "forecast": batch_forecast_df.to_dict(orient="records"),
        "reorder": reorder_df.to_dict(orient="records"),
        "performance": metrics_df.to_dict(orient="records"),
        "volatility": vol_df.to_dict(orient="records"),
        "inventory": sim_df.to_dict(orient="records"),
    }
