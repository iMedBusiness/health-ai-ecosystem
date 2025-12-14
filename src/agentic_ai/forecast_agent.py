# src/agentic_ai/forecast_agent.py

from ai_core.data_pipeline import load_data, preprocess_data
from ai_core.model_training import train_random_forest
import pandas as pd
import numpy as np
import joblib
import os

class ForecastAgent:
    """
    ForecastAgent: Handles end-to-end forecasting for health supply chain
    using Random Forest for demand and lead-time prediction.
    """

    def __init__(self, model_path=None):
        """
        Initialize ForecastAgent.
        If model_path is provided and exists, load the model.
        """
        self.model = None
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"✅ Loaded model from {model_path}")

    def run_pipeline(self, csv_path, date_col, target_col, feature_cols=None, test_size=0.2, random_state=42, save_model_path=None):
        """
        Full pipeline: load → preprocess → train → predict → metrics
        """
        # Step 1: Load data
        df = load_data(csv_path)
        print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

        # Step 2: Preprocess
        df_processed = preprocess_data(df, date_col=date_col, target_col=target_col)
        print(f"✅ Preprocessed data, features: {df_processed.columns.tolist()}")

        # Step 3: Train Random Forest
        model, X_test, y_test, metrics = train_random_forest(
            df_processed,
            target_col='y',
            exclude_cols=("ds",),
            save_model_path=save_model_path
        )

        self.model = model

        # Step 4: Predict on test set
        preds = model.predict(X_test)
        forecast_df = pd.DataFrame({
            'ds': df_processed['ds'].iloc[X_test.index],
            'actual': y_test,
            'forecast': preds
        })

        # Step 5: Metrics
        print(f"✅ Forecast completed. MAE: {metrics['MAE']:.2f}, RMSE: {metrics['RMSE']:.2f}")

        return forecast_df, metrics

    def predict_lead_time(self, X_input):
        """
        Predict lead-time or demand on new input data.
        X_input: pd.DataFrame with same features as training
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Run pipeline first.")
        preds = self.model.predict(X_input)
        return preds