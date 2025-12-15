# src/apps/backend/main.py

from fastapi import FastAPI, UploadFile, File
import pandas as pd

from agentic_ai.forecast_agent import ForecastAgent
from agentic_ai.reorder_agent import ReorderAgent
from ai_core.data_pipeline import preprocess_data

app = FastAPI(
    title="Health AI Supply Chain API",
    description="SaaS-ready Forecasting & Inventory Intelligence",
    version="0.1.0"
)

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Health AI Ecosystem API",
        "version": "0.1.0"
    }


forecast_agent = ForecastAgent()
reorder_agent = ReorderAgent()


@app.post("/forecast")
async def forecast(
    file: UploadFile = File(...),
    horizon: int = 30
):
    """
    Batch demand forecast + reorder points
    """
    # Read CSV
    df = pd.read_csv(file.file)

    # Preprocess
    df_processed = preprocess_data(
        df,
        date_col="date",
        target_col="demand"
    )

    feature_cols = [
        "y",
        "stock_on_hand",
        "day_of_week",
        "month"
    ]

    # Train demand model
    demand_model, _ = forecast_agent.train_demand_model(
        df_processed,
        feature_cols=feature_cols
    )

    # Batch forecast
    forecast_df = forecast_agent.run_batch_forecast(
        df=df_processed,
        model=demand_model,
        periods=horizon
    )

    # Reorder points
    reorder_df = reorder_agent.compute_reorder_point(forecast_df)

    return {
        "forecast": forecast_df.to_dict(orient="records"),
        "reorder_points": reorder_df.to_dict(orient="records")
    }
