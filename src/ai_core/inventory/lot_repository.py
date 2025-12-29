import duckdb
from pathlib import Path

# ---------------------------------------------
# Resolve project root (robust across OS & CWD)
# ---------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

LOTS_PATH = (
    PROJECT_ROOT / "data" / "lakehouse" / "inventory" / "lots"
)


class LotRepository:
    """
    Reads lot-level inventory from Delta Lake
    One row = facility-item-lot
    """

    def __init__(self):
        self.con = duckdb.connect()

    def get_lots(self, facility: str, item: str):
        query = f"""
        SELECT
            facility,
            item,
            lot_id,
            expiry_date,
            qty_on_hand,
            supplier_id
        FROM delta_scan('{LOTS_PATH.as_posix()}')
        WHERE facility = '{facility}'
          AND item = '{item}'
        """
        return self.con.execute(query).df()
