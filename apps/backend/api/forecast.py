# apps/backend/api/forecast.py

from fastapi import APIRouter, HTTPException
import pandas as pd

from apps.backend.schemas.requests import BatchForecastRequest
from src.agentic_ai.forecast_agent import ForecastAgent
from src.agentic_ai.reorder_agent import ReorderAgent
from ai_core.data_pipeline import preprocess_data
from src.ai_core.volatility import classify_volatility

router = APIRouter()


@router.post("/batch")
def batch_forecast(request: BatchForecastRequest):
    """
    Multi-item, multi-facility batch forecast endpoint.
    Fully self-contained: validates, preprocesses, forecasts, reorders.
    """

    # --------------------------------------------------
    # 1️⃣ Load payload into DataFrame
    # --------------------------------------------------
    df_raw = pd.DataFrame(request.data)

    if df_raw.empty:
        raise HTTPException(
            status_code=400,
            detail="Empty dataset received"
        )

    # --------------------------------------------------
    # 2️⃣ Validate required columns
    # --------------------------------------------------
    required_cols = {
        "facility",
        "item",
        request.date_col,
        request.demand_col,
    }

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

    # --------------------------------------------------
    # 3️⃣ Preprocess data (CRITICAL)
    # Creates: ds, y, day_of_week, month
    # --------------------------------------------------
    df = preprocess_data(
        df_raw,
        date_col=request.date_col,
        target_col=request.demand_col,
    )

    # --------------------------------------------------
    # 4️⃣ Demand volatility classification
    # --------------------------------------------------
    try:
        vol_df = classify_volatility(
            df,
            y_col="y",
            group_cols=("facility", "item")
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Volatility classification failed: {str(e)}"
        )

    # --------------------------------------------------
    # 5️⃣ Initialize agents
    # --------------------------------------------------
    forecast_agent = ForecastAgent()
    reorder_agent = ReorderAgent()

    # --------------------------------------------------
    # 6️⃣ Run batch demand forecast
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
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch forecasting failed: {str(e)}",
        )

    # --------------------------------------------------
    # 7️⃣ Prepare lead-time dataframe (optional)
    # --------------------------------------------------
    if "lead_time_days" in df_raw.columns:
        lead_time_df = (
            df_raw
            .groupby(["facility", "item"], as_index=False)
            .agg({"lead_time_days": "mean"})
        )
    else:
        lead_time_df = pd.DataFrame(
            columns=["facility", "item", "lead_time_days"]
        )

    # --------------------------------------------------
    # 8️⃣ Compute reorder points & safety stock
    # --------------------------------------------------
    try:
        reorder_df = reorder_agent.compute_reorder_point(
            forecast_df=batch_forecast_df,
            lead_time_df=lead_time_df,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Reorder computation failed: {str(e)}",
        )

    # --------------------------------------------------
    # 9️⃣ Final response
    # --------------------------------------------------
    return {
        "status": "success",
        "meta": {
            "horizon_days": request.horizon,
            "records_received": len(df_raw),
            "forecast_rows": len(batch_forecast_df),
            "cache_hit_rate": round(metrics_df["cache_hit"].mean(), 2),
            "avg_runtime_sec": round(metrics_df["runtime_sec"].mean(), 2),
        },
        "forecast": batch_forecast_df.to_dict(orient="records"),
        "performance": metrics_df.to_dict(orient="records"),
    }