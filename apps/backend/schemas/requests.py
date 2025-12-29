# apps/backend/schemas/requests.py

from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class BatchForecastRequest(BaseModel):
    # Core data
    data: List[Dict[str, Any]]
    date_col: str
    demand_col: str
    horizon: int
    stock_col: Optional[str] = None

    # 🔐 PERFORMANCE / SCALABILITY CONTROLS
    return_forecast_detail: bool = True
    return_inventory_detail: bool = True
    max_detail_rows: int = 20000

    class Config:
        schema_extra = {
            "example": {
                "date_col": "date",
                "demand_col": "demand",
                "horizon": 30,
                "stock_col": "stock_on_hand",
                "return_forecast_detail": False,
                "return_inventory_detail": False,
                "max_detail_rows": 20000
            }
        }


class ExecutiveSummaryRequest(BaseModel):
    reorder: list
    volatility: list
    inventory_risk: list
    horizon_days: int
