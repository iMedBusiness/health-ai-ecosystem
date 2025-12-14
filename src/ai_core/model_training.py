# src/ai_core/model_training.py

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_random_forest(
    df,
    feature_cols,
    target_col,
    test_size=0.2,
    random_state=42,
    save_model_path=None
):
    """
    Generic Random Forest trainer (used for demand or lead time).
    """

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=random_state
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    if save_model_path:
        joblib.dump(model, save_model_path)

    return model, metrics

def train_lead_time_model(df):
    """
    Train Random Forest model to predict lead time.

    """

    feature_cols = [
        "stock_on_hand",
        "day_of_week",
        "month"
    ]
    target_col = "lead_time_days"
    
    x = df[feature_cols]
    y = df[target_col]
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )
    
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    return model, metrics