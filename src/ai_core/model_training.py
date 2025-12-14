# src/ai_core/model_training.py

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_random_forest(
    df,
    target_col="y",
    exclude_cols=("ds",),
    save_model_path=None
):
    """
    Train Random Forest model with automatic categorical encoding
    """

    # Separate target
    y = df[target_col]

    # Drop excluded + target columns
    X = df.drop(columns=[target_col] + list(exclude_cols))

    # ✅ Encode categorical features automatically
    X = pd.get_dummies(X, drop_first=True)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": rmse,
    }

    # Save model
    if save_model_path:
        joblib.dump(model, save_model_path)
        print(f"✅ Model saved to {save_model_path}")

    return model, X_test, y_test, metrics