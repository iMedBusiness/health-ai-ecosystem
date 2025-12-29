import pandas as pd
from deltalake import write_deltalake

# -----------------------------
# Supplier Pool (Phase 3)
# One row = facility-item-supplier
# -----------------------------

supplier_df = pd.DataFrame([
    {
        "facility": "facility_a",
        "item": "amoxicillin_500mg",
        "supplier_id": "SUP_001",
        "supplier_name": "Local Pharma Co",
        "price_per_unit": 0.12,
        "lead_time_days": 7,
        "lead_time_std": 2,
        "reliability_score": 0.92,
        "capacity_per_period": 50000,
        "min_order_qty": 1000,
        "contracted": True,
        "risk_score": 0.10
    },
    {
        "facility": "facility_a",
        "item": "amoxicillin_500mg",
        "supplier_id": "SUP_002",
        "supplier_name": "Regional Distributor",
        "price_per_unit": 0.10,
        "lead_time_days": 14,
        "lead_time_std": 4,
        "reliability_score": 0.85,
        "capacity_per_period": 100000,
        "min_order_qty": 5000,
        "contracted": False,
        "risk_score": 0.25
    }
])

# Write Delta table
write_deltalake(
    "data/lakehouse/suppliers/supplier_pool",
    supplier_df,
    mode="overwrite"
)

print("✅ supplier_pool Delta table created")
