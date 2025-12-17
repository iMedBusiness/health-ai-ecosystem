import pandas as pd
import numpy as np


class InventorySimulationAgent:
    """
    Simulates inventory position over the forecast horizon and estimates:
    - stockout date
    - days of cover
    - projected end stock
    - reorder recommendation vs ROP
    """

    def simulate(
        self,
        forecast_df,
        inventory_df,
        reorder_df=None,
        date_col="ds",
        demand_col="forecast",
        stock_col="stock_on_hand",
        facility_col="facility",
        item_col="item"
    ):
        """
        forecast_df must contain: facility, item, ds, forecast
        inventory_df must contain: facility, item, stock_on_hand (starting stock)
        reorder_df optional: facility, item, reorder_point
        """

        # Validate columns
        req_forecast = {facility_col, item_col, date_col, demand_col}
        req_inv = {facility_col, item_col, stock_col}

        missing_f = req_forecast - set(forecast_df.columns)
        missing_i = req_inv - set(inventory_df.columns)

        if missing_f:
            raise ValueError(f"forecast_df missing columns: {missing_f}")
        if missing_i:
            raise ValueError(f"inventory_df missing columns: {missing_i}")

        # Merge initial stock into forecast rows
        df = forecast_df.merge(
            inventory_df,
            on=[facility_col, item_col],
            how="left"
        )

        # HARD CHECK
        if stock_col not in df.columns:
            raise ValueError(
                f"'{stock_col}' not found after merge. "
                f"Available columns: {df.columns.tolist()}"
            )
            
        if df[stock_col].isna().all():
            raise ValueError(
                "All stock_on_hand values are NaN after merge. "
                "Check facility/item alignment."
            )

        # If reorder_df provided, merge reorder point
        if reorder_df is not None:
            df = df.merge(
                reorder_df[[facility_col, item_col, "reorder_point"]],
                on=[facility_col, item_col],
                how="left"
            )
        else:
            df["reorder_point"] = np.nan

        # Ensure date ordering
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([facility_col, item_col, date_col])

        results = []

        for (facility, item), g in df.groupby([facility_col, item_col]):
            g = g.copy()

            start_stock = g[stock_col].iloc[0]
            if pd.isna(start_stock):
                continue

            # Daily consumption assumed = forecast demand
            g["projected_stock"] = start_stock - g[demand_col].cumsum()

            # Stockout detection
            stockout_rows = g[g["projected_stock"] <= 0]

            if len(stockout_rows) > 0:
                stockout_date = stockout_rows[date_col].iloc[0]
                days_of_cover = (stockout_date - g[date_col].iloc[0]).days
            else:
                stockout_date = None
                days_of_cover = len(g)

            end_stock = g["projected_stock"].iloc[-1]
            reorder_point = g["reorder_point"].iloc[0]

            # Reorder recommendation (simple rule)
            reorder_now = False
            reorder_qty = None

            # If reorder point is known, compare current stock to ROP
            if not pd.isna(reorder_point):
                if start_stock <= reorder_point:
                    reorder_now = True
                    reorder_qty = max(reorder_point - start_stock, 0)

            results.append({
                "facility": facility,
                "item": item,
                "start_stock": round(float(start_stock), 2),
                "end_stock": round(float(end_stock), 2),
                "days_of_cover": int(days_of_cover),
                "stockout_date": stockout_date.strftime("%Y-%m-%d") if stockout_date else None,
                "reorder_point": round(float(reorder_point), 2) if not pd.isna(reorder_point) else None,
                "reorder_now": reorder_now,
                "recommended_reorder_qty": round(float(reorder_qty), 2) if reorder_qty is not None else None
            })

        return pd.DataFrame(results)
