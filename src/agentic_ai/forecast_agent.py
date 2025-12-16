# src/agentic_ai/forecast_agent.py

from ai_core.data_pipeline import load_data, preprocess_data
from ai_core.model_training import train_random_forest
from ai_core.future_forecast import forecast_future_demand
from ai_core.explainability import (
    compute_shap_values,
    plot_global_importance
)
import shap
import pandas as pd

class ForecastAgent:
    """
    Agent responsible for training forecasting-related models.
    """
    def train_demand_model(self, df, feature_cols):
        from ai_core.model_training import train_random_forest

        model, metrics = train_random_forest(
            df=df,
            feature_cols=feature_cols,
            target_col="y"
        )
        return model, metrics

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

        # Train lead-time RF model
        model, metrics = train_random_forest(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            save_model_path=save_model_path
        )

        print("✅ Lead-time RF trained")
        print("📊 Metrics:", metrics)

        # SHAP EXPLAINABILITY
        X_sample = df[feature_cols].head(50)
        explainer, shap_values = compute_shap_values(model=model, X=X_sample)
        
        # Store for Streamlit & reasoning
        self.lead_time_shap = {"explainer": explainer,"shap_values":shap_values, "X_sample": X_sample, "feature_cols": feature_cols}
        print("🔍 SHAP explainability ready")
        
        return model, metrics
    
    def run_future_forecast(
        self,
        model,
        df_processed,
        periods=30
    ):
        feature_cols = [
            "y",
            "stock_on_hand",
            "day_of_week",
            "month"
        ]

        forecast_df = forecast_future_demand(
            model=model,
            df_history=df_processed,
            feature_cols=feature_cols,
            periods=periods
        )

        print(f"✅ Generated {periods}-day demand forecast")
        return forecast_df
    
    def run_batch_forecast(
        self,
        df,
        item_col="item",
        facility_col="facility",
        date_col="ds",
        target_col="y",
        model=None,
        periods=30
    ):
        """
        Run forecasting across multiple items and facilities.
        Returns a concatenated DataFrame with forecasts.
        """
        forecast_list = []

        unique_items = df[item_col].unique()
        unique_facilities = df[facility_col].unique()

        for facility in unique_facilities:
            for item in unique_items:
                df_subset = df[(df[facility_col] == facility) & (df[item_col] == item)]
                if df_subset.empty:
                    continue

                forecast_df = self.run_future_forecast(
                    model=model,
                    df_processed=df_subset,
                    periods=periods
                )
                forecast_df[facility_col] = facility
                forecast_df[item_col] = item
                forecast_df["forecast_horizon_days"] = periods
                forecast_df["model_type"] = "random_forest"

                forecast_list.append(forecast_df)

        if forecast_list:
            return pd.concat(forecast_list, axis=0).reset_index(drop=True)
        else:
            return pd.DataFrame()

    def compute_shap(self, model, df, feature_cols, sample_size=100):
        """
        Compute SHAP values for a trained model.
        Returns:
            explainer: SHAP explainer object
            shap_values: SHAP values array
            X_sample: sample of features used
        """
        # Take a sample to speed up SHAP computation
        X_sample = df[feature_cols].sample(min(sample_size, len(df)), random_state=42)
        
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        
        return explainer, shap_values, X_sample
    
    
