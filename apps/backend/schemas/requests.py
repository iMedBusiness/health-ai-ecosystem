# apps/backend/schemas/requests.py

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class BatchForecastRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_col: str 
    demand_col: str 
    horizon: int 
    stock_col: Optional[str] = None

class ExecutiveSummaryRequest(BaseModel):
    reorder: list
    volatility: list
    inventory_risk: list
    horizon_days: int