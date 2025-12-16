import pandas as pd


class ExplainableReorderAgent:
    """
    Generates human-readable explanations
    for reorder point drivers.
    """

    def explain_reorder_drivers(self, reorder_df):
        """
        reorder_df must contain:
        - avg_daily_demand
        - lead_time_days
        - safety_stock
        - reorder_point
        """

        explanations = []

        # Benchmarks for relative comparison
        demand_med = reorder_df["avg_daily_demand"].median()
        lead_time_med = reorder_df["lead_time_days"].median()
        safety_med = reorder_df["safety_stock"].median()

        for _, row in reorder_df.iterrows():
            drivers = []

            if row["lead_time_days"] > lead_time_med:
                drivers.append("longer-than-average lead time")

            if row["avg_daily_demand"] > demand_med:
                drivers.append("higher-than-average demand")

            if row["safety_stock"] > safety_med:
                drivers.append("high demand variability (safety stock)")

            if not drivers:
                drivers.append("stable demand and lead time")

            explanation = (
                f"**{row['item']} @ {row['facility']}** — "
                f"Reorder point is elevated due to "
                + ", ".join(drivers)
                + "."
            )

            explanations.append(explanation)

        return explanations

    def compute_driver_scores(self, reorder_df):
        """
        Returns normalized contribution scores
        for explainability dashboards.
        """

        df = reorder_df.copy()

        df["demand_score"] = (
            df["avg_daily_demand"] / df["avg_daily_demand"].max()
        )

        df["lead_time_score"] = (
            df["lead_time_days"] / df["lead_time_days"].max()
        )

        df["variability_score"] = (
            df["safety_stock"] / df["safety_stock"].max()
        )

        return df[
            [
                "facility",
                "item",
                "demand_score",
                "lead_time_score",
                "variability_score"
            ]
        ]