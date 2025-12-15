import pandas as pd
import numpy as np

class ReorderAgent:
    """
    Computes Reorder Points (ROP) and Safety Stock (SS) for multiple items/facilities.
    """
    def compute_reorder_point(
        self,
        forecast_df,
        lead_time_col="lead_time_days",
        demand_col="forecast",
        service_level=1.65  # 95% service level ~ 1.65 z-score
    ):
        """
        safety_stock = z * std_dev_demand * sqrt(lead_time)
        reorder_point = avg_demand * lead_time + safety_stock
        """
        reorder_list = []

        grouped = forecast_df.groupby(["facility", "item"])
        for (facility, item), group in grouped:
            avg_demand = group[demand_col].mean()
            std_demand = group[demand_col].std()
            lead_time = group[lead_time_col].mean()

            safety_stock = service_level * std_demand * np.sqrt(lead_time)
            reorder_point = avg_demand * lead_time + safety_stock

            reorder_list.append({
                "facility": facility,
                "item": item,
                "safety_stock": round(safety_stock, 2),
                "reorder_point": round(reorder_point, 2)
            })

        return pd.DataFrame(reorder_list)
