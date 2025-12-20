# apps/backend/schemas/requests.py

from pydantic import BaseModel
from typing import List, Dict, Any

class BatchForecastRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_col: str = "date"
    demand_col: str = "demand"
    horizon: int = 30

class ExecutiveSummaryRequest(BaseModel):
    reorder: List[Dict]
    horizon_days: int