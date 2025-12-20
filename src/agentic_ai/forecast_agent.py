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
from concurrent.futures import ThreadPoolExecutor, as_completed
from ai_core.model_cache import (
    load_cached_model,
    save_cached_model
)

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
        periods=30,
        cache_dir="models/cache",
        force_retrain=False,
        parallel=True,
        max_workers=4
    ):
        from ai_core.model_training import train_random_forest
        from ai_core.future_forecast import forecast_future_demand
        from ai_core.model_cache import load_cached_model, save_cached_model
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import pandas as pd
        import time

        required_cols = {"facility", "item", "ds", "y", "day_of_week", "month"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        feature_cols = ["day_of_week", "month"]

        groups = []
        for (facility, item), g in df.groupby(["facility", "item"]):
            g = g.sort_values("ds")
            if len(g) >= 5:
                groups.append((facility, item, g))

        if not groups:
            raise ValueError("No valid facility–item combinations")

        metrics = []
        results = []

        def _worker(facility, item, g):
            start = time.time()
            model_name = "rf_demand"
            cache_hit = False

            model = None
            if not force_retrain:
                model, _ = load_cached_model(cache_dir, facility, item, model_name)
                cache_hit = model is not None

            if model is None:
                model, _ = train_random_forest(
                    df=g,
                    feature_cols=feature_cols,
                    target_col="y"
                )
                save_cached_model(model, cache_dir, facility, item, model_name)

            future_df = forecast_future_demand(
                model=model,
                df_history=g,
                feature_cols=feature_cols,
                periods=periods
            )
            future_df["facility"] = facility
            future_df["item"] = item

            duration = round(time.time() - start, 3)

            metric = {
                "facility": facility,
                "item": item,
                "cache_hit": cache_hit,
                "runtime_sec": duration
            }

            return future_df, metric

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(_worker, f, i, g)
                    for f, i, g in groups
                ]
                for f in as_completed(futures):
                    forecast_df, metric = f.result()
                    results.append(forecast_df)
                    metrics.append(metric)
        else:
            for f, i, g in groups:
                forecast_df, metric = _worker(f, i, g)
                results.append(forecast_df)
                metrics.append(metric)

        return {
            "forecast": pd.concat(results, ignore_index=True),
            "metrics": pd.DataFrame(metrics)
        }

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
    
    
