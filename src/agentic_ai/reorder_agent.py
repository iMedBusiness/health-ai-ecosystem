# src/agentic_ai/reorder_agent.py

import pandas as pd
import numpy as np


class ReorderAgent:
    """
    Computes Reorder Points (ROP) and Safety Stock (SS)
    using forecasted demand and embedded lead time.
    """

    def compute_reorder_point(
        self,
        forecast_df: pd.DataFrame,
        demand_col: str = "forecast",
        lead_time_col: str = "lead_time_days",
        service_level_z: float = 1.65,
    ) -> pd.DataFrame:

        required = {"facility", "item", demand_col, lead_time_col}
        missing = required - set(forecast_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        results = []

        for (facility, item), g in forecast_df.groupby(["facility", "item"]):
            avg_demand = g[demand_col].mean()
            std_demand = g[demand_col].std() or 0.0
            lead_time = g[lead_time_col].mean()

            if pd.isna(lead_time) or lead_time <= 0:
                continue

            safety_stock = service_level_z * std_demand * (lead_time ** 0.5)
            reorder_point = avg_demand * lead_time + safety_stock

            results.append({
                "facility": facility,
                "item": item,
                "avg_daily_demand": round(avg_demand, 2),
                "lead_time_days": round(lead_time, 1),
                "safety_stock": round(safety_stock, 2),
                "reorder_point": round(reorder_point, 2),
            })

        return pd.DataFrame(results)
