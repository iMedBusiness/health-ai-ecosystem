import duckdb
from pathlib import Path

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SUPPLIER_POOL_PATH = (
    PROJECT_ROOT / "data" / "lakehouse" / "suppliers" / "supplier_pool"
)

class SupplierRepository:
    def __init__(self):
        self.con = duckdb.connect()

    def get_suppliers(self, facility: str, item: str):
        query = f"""
        SELECT
            facility,
            item,
            supplier_id,
            supplier_name,
            price_per_unit,
            lead_time_days,
            lead_time_std,
            reliability_score,
            capacity_per_period,
            min_order_qty,
            contracted,
            risk_score
        FROM delta_scan('{SUPPLIER_POOL_PATH.as_posix()}')
        WHERE facility = '{facility}'
          AND item = '{item}'
        """
        return self.con.execute(query).df()
