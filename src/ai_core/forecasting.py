import pandas as pd
from datetime import timedelta

def forecast_future(model, last_date, periods, base_features):
    """
    Stub: future forecasting logic will be added later
    """
    future_dates = [
        last_date + timedelta(days=i+1) for i in range(periods)
    ]

    return pd.DataFrame({
        "ds": future_dates,
        "forecast": [None] * periods
    })
