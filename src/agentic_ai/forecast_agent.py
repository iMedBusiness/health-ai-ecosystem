# src/agentic_ai/forecast_agent.py

from ai_core.data_pipeline import load_data, preprocess_data
from ai_core.model_training import train_random_forest

class ForecastAgent:
    """
    Agent responsible for training forecasting-related models.
    """

    def run_lead_time_model(
        self,
        csv_path,
        date_col,
        target_col="lead_time_days",
        save_model_path=None
    ):
        # Load data
        df = load_data(csv_path)
        print(f"✅ Loaded data: {df.shape}")

        # Preprocess
        df = preprocess_data(
            df,
            date_col=date_col,
            target_col="demand"
        )

        feature_cols = [
            "y",
            "stock_on_hand",
            "day_of_week",
            "month"
        ]

        model, metrics = train_random_forest(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            save_model_path=save_model_path
        )

        print("✅ Lead-time RF trained")
        print("📊 Metrics:", metrics)

        return model, metrics