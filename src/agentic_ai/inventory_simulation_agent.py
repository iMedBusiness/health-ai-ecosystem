# src/agentic_ai/inventory_simulation_agent.py

from __future__ import annotations
import pandas as pd
import numpy as np

class InventorySimulationAgent:
    """
    Simulates daily inventory evolution per facility-item given:
      - forecast_df: columns [facility, item, ds, forecast, lead_time_days]
      - inventory_df: columns [facility, item, stock_on_hand]
      - reorder_df: columns [facility, item, reorder_point, avg_daily_demand, lead_time_days]

    Assumptions:
      - Demand each day = forecast (can be fractional)
      - Reorder decision is made daily using inventory_position
      - Orders arrive after lead_time_days (rounded to int days)
      - Order quantity = order_up_to_days * avg_daily_demand (simple order-up-to policy)
    """

    def simulate(
        self,
        forecast_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        reorder_df: pd.DataFrame,
        stock_col: str = "stock_on_hand",
        demand_col: str = "forecast",
        date_col: str = "ds",
        lead_time_col: str = "lead_time_days",
        reorder_point_col: str = "reorder_point",
        avg_demand_col: str = "avg_daily_demand",
        order_up_to_days: int = 14,
        min_order_qty: float = 0.0,
    ) -> pd.DataFrame:

        # ---------- Validate ----------
        req_f = {"facility", "item", date_col, demand_col}
        missing = req_f - set(forecast_df.columns)
        if missing:
            raise ValueError(f"forecast_df missing columns: {missing}")

        req_i = {"facility", "item", stock_col}
        missing = req_i - set(inventory_df.columns)
        if missing:
            raise ValueError(f"inventory_df missing columns: {missing}")

        req_r = {"facility", "item", reorder_point_col, avg_demand_col}
        missing = req_r - set(reorder_df.columns)
        if missing:
            raise ValueError(f"reorder_df missing columns: {missing}")

        # ---------- Normalize types ----------
        f = forecast_df.copy()
        i = inventory_df.copy()
        r = reorder_df.copy()

        for df in (f, i, r):
            for c in ["facility", "item"]:
                df[c] = df[c].astype(str).str.strip().str.lower()

        f[date_col] = pd.to_datetime(f[date_col], errors="coerce")
        f = f.dropna(subset=[date_col]).sort_values([ "facility", "item", date_col ])

        i[stock_col] = pd.to_numeric(i[stock_col], errors="coerce").fillna(0.0)
        r[reorder_point_col] = pd.to_numeric(r[reorder_point_col], errors="coerce")
        r[avg_demand_col] = pd.to_numeric(r[avg_demand_col], errors="coerce")

        # Lead time: prefer reorder_df lead_time, else forecast_df lead_time, else default 7
        if lead_time_col in r.columns:
            r[lead_time_col] = pd.to_numeric(r[lead_time_col], errors="coerce")
        if lead_time_col in f.columns:
            f[lead_time_col] = pd.to_numeric(f[lead_time_col], errors="coerce")

        # ---------- Join setup ----------
        # attach reorder params to each forecast row
        f = f.merge(
            r[["facility","item", reorder_point_col, avg_demand_col] + ([lead_time_col] if lead_time_col in r.columns else [])],
            on=["facility","item"],
            how="left"
        )

        # attach starting stock
        f = f.merge(
            i[["facility","item", stock_col]],
            on=["facility","item"],
            how="left"
        )

        if stock_col not in f.columns:
            # No inventory information available → assume zero stock
            f[stock_col] = 0.0
        else:
            f[stock_col] = pd.to_numeric(
            f[stock_col], errors="coerce"
            ).fillna(0.0)
        f[reorder_point_col] = f[reorder_point_col].fillna(np.nan)
        f[avg_demand_col] = f[avg_demand_col].fillna(np.nan)

        # if lead_time missing, take from forecast column, else default
        if lead_time_col in f.columns:
            f[lead_time_col] = f[lead_time_col].fillna(7.0)
        else:
            f[lead_time_col] = 7.0

        # ---------- Simulation ----------
        rows = []

        for (facility, item), g in f.groupby(["facility", "item"], sort=False):
            g = g.sort_values(date_col).reset_index(drop=True)

            on_hand = float(g.loc[0, stock_col])
            on_order = []  # list of tuples: (arrival_date, qty)

            rp = g[reorder_point_col].dropna()
            rp = float(rp.iloc[0]) if len(rp) else np.nan

            avgd = g[avg_demand_col].dropna()
            avgd = float(avgd.iloc[0]) if len(avgd) else np.nan

            # If we can't compute policy params, still simulate consumption (no ordering)
            can_order = (not np.isnan(rp)) and (not np.isnan(avgd)) and avgd > 0

            for t in range(len(g)):
                d = g.loc[t, date_col]
                demand = float(g.loc[t, demand_col]) if pd.notna(g.loc[t, demand_col]) else 0.0
                lt = g.loc[t, lead_time_col]
                lt_days = int(max(0, round(float(lt)))) if pd.notna(lt) else 7

                # 1) Receive orders arriving today
                if on_order:
                    arriving = [x for x in on_order if x[0] <= d]
                    if arriving:
                        recv_qty = sum(q for _, q in arriving)
                        on_hand += recv_qty
                        on_order = [x for x in on_order if x[0] > d]

                # 2) Consume demand
                on_hand = max(0.0, on_hand - demand)

                # 3) Compute inventory position
                outstanding = sum(q for _, q in on_order) if on_order else 0.0
                inv_position = on_hand + outstanding

                # 4) Reorder decision
                reorder_now = False
                order_qty = 0.0
                arrival_date = pd.NaT

                if can_order and inv_position <= rp:
                    reorder_now = True
                    # simple order-up-to: cover N days of demand
                    order_qty = max(min_order_qty, avgd * float(order_up_to_days))
                    arrival_date = d + pd.Timedelta(days=lt_days)
                    if order_qty > 0:
                        on_order.append((arrival_date, order_qty))

                # 5) Days of cover (based on avg demand param, fallback to demand)
                denom = avgd if (pd.notna(avgd) and avgd > 0) else max(demand, 1e-6)
                days_of_cover = on_hand / denom

                rows.append({
                    "facility": facility,
                    "item": item,
                    "ds": d,
                    "forecast": demand,
                    "stock_on_hand": round(on_hand, 2),
                    "inventory_position": round(inv_position, 2),
                    "days_of_cover": round(days_of_cover, 2),
                    "reorder_point": None if np.isnan(rp) else round(rp, 2),
                    "lead_time_days": lt_days,
                    "reorder_now": bool(reorder_now),
                    "order_qty": round(order_qty, 2),
                    "order_arrival_ds": arrival_date if pd.notna(arrival_date) else None,
                    "outstanding_orders_qty": round(outstanding, 2),
                })

        return pd.DataFrame(rows)
