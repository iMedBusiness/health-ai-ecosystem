import pandas as pd
from pathlib import Path
from deltalake import write_deltalake

# -------------------------------------------------
# Resolve project root safely (same pattern as before)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOTS_PATH = (
    PROJECT_ROOT / "data" / "lakehouse" / "inventory" / "lots"
)

LOTS_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Mock lot-level inventory data (Phase 4)
# One row = facility-item-lot
# -------------------------------------------------
lots_df = pd.DataFrame([
    {
        "facility": "facility_a",
        "item": "amoxicillin_500mg",
        "lot_id": "LOT_001",
        "expiry_date": "2026-01-10",
        "qty_on_hand": 8000,
        "supplier_id": "SUP_001"
    },
    {
        "facility": "facility_a",
        "item": "amoxicillin_500mg",
        "lot_id": "LOT_002",
        "expiry_date": "2026-02-01",
        "qty_on_hand": 12000,
        "supplier_id": "SUP_001"
    },
    {
        "facility": "facility_a",
        "item": "amoxicillin_500mg",
        "lot_id": "LOT_003",
        "expiry_date": "2026-04-15",
        "qty_on_hand": 20000,
        "supplier_id": "SUP_002"
    }
])

# -------------------------------------------------
# Write Delta table
# -------------------------------------------------
write_deltalake(
    LOTS_PATH.as_posix(),
    lots_df,
    mode="overwrite"
)

print("✅ Inventory lots Delta table created at:")
print(LOTS_PATH)
