import pandas as pd

class InventoryRiskAgent:
    """
    Assigns inventory risk level (LOW / MEDIUM / HIGH)
    based on demand, volatility, and stock position.
    """

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {
            "facility",
            "item",
            "days_of_cover",
            "reorder_now",
            "volatility_class",
        }

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for risk scoring: {missing}")

        def classify(row):
            if (
                row["reorder_now"] is True
                or row["days_of_cover"] < 7
                or row["volatility_class"] == "Erratic"
            ):
                return "HIGH"
            if pd.isna(row["days_of_cover"]):
                return "HIGH"

            if (
                row["days_of_cover"] < 14
                or row["volatility_class"] == "Seasonal"
            ):
                return "MEDIUM"

            return "LOW"

        df = df.copy()
        df["inventory_risk"] = df.apply(classify, axis=1)
        return df
