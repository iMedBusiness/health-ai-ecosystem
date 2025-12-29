import pandas as pd

DEFAULT_WEIGHTS = {
    "price": 0.30,
    "lead_time": 0.30,
    "reliability": 0.30,
    "risk": 0.10
}

class SupplierRanker:

    def __init__(self, weights=None):
        self.weights = weights or DEFAULT_WEIGHTS

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["price_norm"] = df["price_per_unit"] / df["price_per_unit"].max()

        df["lead_time_norm"] = (
            df["lead_time_days"] + df["lead_time_std"]
        ) / (df["lead_time_days"] + df["lead_time_std"]).max()

        df["reliability_risk"] = 1 - df["reliability_score"]
        df["risk_norm"] = df["risk_score"]

        df["supplier_score"] = (
            self.weights["price"] * df["price_norm"]
            + self.weights["lead_time"] * df["lead_time_norm"]
            + self.weights["reliability"] * df["reliability_risk"]
            + self.weights["risk"] * df["risk_norm"]
        )

        df["rank"] = df["supplier_score"].rank(method="dense")

        return df.sort_values("supplier_score")
