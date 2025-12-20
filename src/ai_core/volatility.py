import pandas as pd
import numpy as np

def classify_volatility(df: pd.DataFrame, y_col="y", group_cols=("facility","item")) -> pd.DataFrame:
    """
    Returns dataframe with facility,item, mean, std, cv, volatility_label
    """
    g = df.groupby(list(group_cols))[y_col].agg(["mean","std"]).reset_index()
    g["mean"] = g["mean"].astype(float)
    g["std"] = g["std"].fillna(0).astype(float)
    g["cv"] = np.where(g["mean"] <= 0, np.nan, g["std"] / g["mean"])

    def label(cv):
        if pd.isna(cv):
            return "Unknown"
        if cv < 0.25:
            return "Low"
        if cv <= 0.60:
            return "Medium"
        return "High"

    g["volatility"] = g["cv"].apply(label)
    return g
