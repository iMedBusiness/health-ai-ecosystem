import pandas as pd


class FEFOAllocator:
    """
    First-Expire, First-Out allocation logic.
    """

    def allocate(self, lots_df: pd.DataFrame, required_qty: float) -> pd.DataFrame:
        if lots_df.empty or required_qty <= 0:
            return pd.DataFrame([{
                "lot_id": None,
                "expiry_date": None,
                "allocated_qty": 0.0,
                "status": "NO_STOCK"
            }])

        df = lots_df.copy()

        # Ensure proper types
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
        df["qty_on_hand"] = df["qty_on_hand"].astype(float)

        # FEFO: earliest expiry first
        df = df.sort_values("expiry_date")

        remaining_qty = float(required_qty)
        allocations = []

        for _, row in df.iterrows():
            if remaining_qty <= 0:
                break

            available = float(row["qty_on_hand"])
            if available <= 0:
                continue

            take_qty = min(available, remaining_qty)

            allocations.append({
                "lot_id": row["lot_id"],
                "expiry_date": (
                    row["expiry_date"].date().isoformat()
                    if pd.notnull(row["expiry_date"])
                    else None
                ),
                "allocated_qty": round(take_qty, 2),
                "status": "ALLOCATED"
            })

            remaining_qty -= take_qty

        if remaining_qty > 0:
            allocations.append({
                "lot_id": "UNFULFILLED",
                "expiry_date": None,
                "allocated_qty": round(remaining_qty, 2),
                "status": "INSUFFICIENT_STOCK"
            })

        return pd.DataFrame(allocations)
