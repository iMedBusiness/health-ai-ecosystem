import pandas as pd
import numpy as np

class ReorderAgent:
    """
    Computes Reorder Points (ROP) and Safety Stock (SS) for multiple items/facilities.
    """
    def compute_reorder_point(
        self,
        forecast_df,
        lead_time_df,
        demand_col="forecast",
        lead_time_col="lead_time_days",
        service_level_z=1.65
    ):
        """
        Computes safety stock and reorder point using:
        - forecasted demand
        - historical average lead time
        """

        results = []

        # Merge lead time into forecast
        df = forecast_df.merge(
            lead_time_df,
            on=["facility", "item"],
            how="left"
        )

        if lead_time_col not in df.columns:
            raise ValueError("lead_time_days not found after merge")

        for (facility, item), g in df.groupby(["facility", "item"]):
            avg_demand = g[demand_col].mean()
            std_demand = g[demand_col].std(ddof=0) or 0.0
            lead_time = g[lead_time_col].mean()

            if pd.isna(lead_time):
                continue

            safety_stock = service_level_z * std_demand * (lead_time ** 0.5)
            reorder_point = avg_demand * lead_time + safety_stock

            results.append({
                "facility": facility,
                "item": item,
                "avg_daily_demand": round(avg_demand, 2),
                "lead_time_days": round(lead_time, 1),
                "safety_stock": round(safety_stock, 2),
                "reorder_point": round(reorder_point, 2)
            })

        return pd.DataFrame(results)
