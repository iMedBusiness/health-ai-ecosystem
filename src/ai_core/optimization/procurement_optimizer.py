from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pandas as pd
import pulp


@dataclass
class OptimizationConfig:
    mode: str = "normal"  # "normal" or "emergency"

    # Exposure cap (concentration control)
    max_share_normal: float = 0.70
    max_share_emergency: float = 0.50

    # Penalties / economics
    shortage_penalty_per_unit: float = 5.0        # tune per item criticality
    expiry_penalty_rate: float = 0.25             # expected loss fraction of value when at-risk

    # Weighting (single optimizer, different weights)
    weight_procurement: float = 1.0
    weight_expiry: float = 1.0
    weight_shortage: float = 1.0


class ProcurementOptimizer:
    """
    MILP optimizer:
    - decision vars: x_s (order quantity), y_s (binary: whether supplier is used)
    - objective: procurement + expiry_penalty + shortage_penalty
    - constraints: demand (soft), capacity, MOQ enforced by y_s, exposure cap
    """

    def optimize(
        self,
        suppliers_df: pd.DataFrame,
        required_qty: float,
        pct_at_risk_90: float,
        config: Optional[OptimizationConfig] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        if config is None:
            config = OptimizationConfig()

        df = suppliers_df.copy()
        if df.empty:
            raise ValueError("No suppliers provided to optimizer.")

        # Basic validation
        required_qty = float(required_qty)
        pct_at_risk_90 = float(pct_at_risk_90)

        # Determine max share cap
        max_share = config.max_share_emergency if config.mode == "emergency" else config.max_share_normal

        # Create MILP model
        model = pulp.LpProblem("ProcurementOptimization", pulp.LpMinimize)

        # Decision variables
        x = {}  # order quantity per supplier
        y = {}  # binary use flag (for MOQ enforcement)

        for i, r in df.iterrows():
            sid = r["supplier_id"]
            x[sid] = pulp.LpVariable(f"x_{sid}", lowBound=0, cat="Continuous")
            y[sid] = pulp.LpVariable(f"y_{sid}", lowBound=0, upBound=1, cat="Binary")

        # Shortage variable (soft demand)
        shortage = pulp.LpVariable("shortage", lowBound=0, cat="Continuous")

        # ------------------------
        # Constraints
        # ------------------------

        # Soft demand constraint: sum(x) + shortage >= required
        model += pulp.lpSum(x.values()) + shortage >= required_qty, "DemandSoft"

        # Per-supplier constraints: capacity, MOQ with binary, exposure cap
        for i, r in df.iterrows():
            sid = r["supplier_id"]
            cap = float(r["capacity_per_period"])
            moq = float(r["min_order_qty"])

            # Capacity: x_s <= cap
            model += x[sid] <= cap, f"Cap_{sid}"

            # Exposure cap: x_s <= max_share * required
            model += x[sid] <= max_share * required_qty, f"ShareCap_{sid}"

            # MOQ enforced by binary:
            # x_s >= MOQ * y_s
            model += x[sid] >= moq * y[sid], f"MOQmin_{sid}"

            # x_s <= cap * y_s (prevents x when y=0)
            model += x[sid] <= cap * y[sid], f"MOQlink_{sid}"

        # ------------------------
        # Objective components
        # ------------------------

        procurement_cost = pulp.lpSum(
            x[r["supplier_id"]] * float(r["price_per_unit"])
            for _, r in df.iterrows()
        )

        # Expiry penalty proxy: proportional to value * pct_at_risk_90
        expiry_cost = pulp.lpSum(
            x[r["supplier_id"]] * float(r["price_per_unit"]) * config.expiry_penalty_rate * pct_at_risk_90
            for _, r in df.iterrows()
        )

        shortage_cost = shortage * config.shortage_penalty_per_unit

        # Weighted single-objective
        model += (
            config.weight_procurement * procurement_cost
            + config.weight_expiry * expiry_cost
            + config.weight_shortage * shortage_cost
        ), "TotalCost"

        # Solve
        status = model.solve(pulp.PULP_CBC_CMD(msg=False))

        # Collect results
        sol = []
        for _, r in df.iterrows():
            sid = r["supplier_id"]
            qty = float(pulp.value(x[sid]) or 0.0)
            used = int(round(float(pulp.value(y[sid]) or 0.0)))
            if qty > 0:
                sol.append({
                    "supplier_id": sid,
                    "supplier_name": r.get("supplier_name", ""),
                    "ordered_qty": round(qty, 2),
                    "used_flag": used,
                    "price_per_unit": float(r["price_per_unit"])
                })

        shortage_val = float(pulp.value(shortage) or 0.0)

        meta = {
            "status": pulp.LpStatus[status],
            "mode": config.mode,
            "required_qty": required_qty,
            "shortage": round(shortage_val, 2),
            "pct_at_risk_90": round(pct_at_risk_90, 4),
            "objective_value": round(float(pulp.value(model.objective) or 0.0), 4)
        }

        return pd.DataFrame(sol).sort_values("ordered_qty", ascending=False), meta
