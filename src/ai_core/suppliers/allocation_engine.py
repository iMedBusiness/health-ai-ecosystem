import pandas as pd

NORMAL_MAX_SHARE = 0.70
EMERGENCY_MAX_SHARE = 0.50


class AllocationEngine:

    def allocate(
        self,
        ranked_df: pd.DataFrame,
        required_qty: float,
        mode: str = "normal"
    ) -> pd.DataFrame:

        df = ranked_df.copy()

        if df.empty:
            raise ValueError("No suppliers available for allocation")

        max_share = (
            EMERGENCY_MAX_SHARE if mode == "emergency"
            else NORMAL_MAX_SHARE
        )

        # Convert supplier_score → allocation weight
        # lower score = better supplier
        df["weight"] = 1 / df["supplier_score"]
        df["weight"] = df["weight"] / df["weight"].sum()

        remaining_qty = required_qty
        allocations = []

        for _, row in df.iterrows():

            supplier_cap = min(
                row["capacity_per_period"],
                required_qty * max_share
            )

            proposed_qty = required_qty * row["weight"]
            allocated_qty = min(proposed_qty, supplier_cap, remaining_qty)

            # Respect MOQ
            if allocated_qty < row["min_order_qty"]:
                continue

            allocations.append({
                "supplier_id": row["supplier_id"],
                "supplier_name": row["supplier_name"],
                "allocated_qty": round(allocated_qty, 2),
                "supplier_score": row["supplier_score"],
                "mode": mode
            })

            remaining_qty -= allocated_qty

            if remaining_qty <= 0:
                break

        if remaining_qty > 0:
            allocations.append({
                "supplier_id": "UNALLOCATED",
                "supplier_name": "Residual Risk",
                "allocated_qty": round(remaining_qty, 2),
                "supplier_score": None,
                "mode": mode
            })

        return pd.DataFrame(allocations)
