from typing import Dict, List

DATASET_SPECS: Dict[str, List[str]] = {

    "items": [
        "item_id",
        "item_code",
        "item_name",
        "category",
        "unit",
        "cold_chain_required",
        "hazardous",
        "essential_medicine_flag",
    ],

    "facilities": [
        "facility_id",
        "facility_name",
        "facility_type",
        "location_id",
    ],

    "lots": [
        "lot_id",
        "item_id",
        "batch_no",
        "expiry_date",
        "received_date",
    ],

    "inventory_balance": [
        "lot_id",
        "facility_id",
        "quantity_on_hand",
        "status",
    ],

    "consumption": [
        "facility_id",
        "item_id",
        "service_date",
        "qty_used",
    ],

    "forecast_daily": [
        "facility_id",
        "item_id",
        "forecast_period",
        "forecast_qty",
    ],
}
