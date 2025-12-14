import pandas as pd

def generate_future_dataframe(last_date, periods):
    """
    Generate future date dataframe.
    """
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=periods,
        freq="D"
    )

    df_future = pd.DataFrame({"ds": future_dates})
    df_future["day_of_week"] = df_future["ds"].dt.dayofweek
    df_future["month"] = df_future["ds"].dt.month

    return df_future


def forecast_future_demand(
    model,
    df_history,
    feature_cols,
    periods=30
):
    """
    Forecast demand for future periods using trained RF model.
    """

    last_date = df_history["ds"].max()

    df_future = generate_future_dataframe(last_date, periods)

    # Use last known values for lag-style features
    df_future["y"] = df_history["y"].iloc[-1]
    df_future["stock_on_hand"] = df_history["stock_on_hand"].iloc[-1]

    X_future = df_future[feature_cols]
    df_future["forecast"] = model.predict(X_future)

    return df_future