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
        df = df.copy()
        for col in required:
            if col not in df.columns:
                df[col] = None

        def classify(row):

            # Fail-safe: missing coverage is risky
            if pd.isna(row.get("days_of_cover")):
                return "HIGH"

            # Safe boolean check (numpy.bool_ aware)
            if pd.notna(row.get("reorder_now")) and bool(row.get("reorder_now")):
                return "HIGH"

            vc = str(row.get("volatility_class", "")).strip().lower()

            # Severe risk conditions
            if row["days_of_cover"] <= 3:
                return "HIGH"

            if vc == "erratic":
                return "HIGH"

            # Medium risk conditions
            if row["days_of_cover"] <= 7 or vc == "seasonal":
                return "MEDIUM"

            # Default: structurally safe
            return "LOW"


        
        df = df.copy()
        df["inventory_risk"] = df.apply(classify, axis=1)
        return df
