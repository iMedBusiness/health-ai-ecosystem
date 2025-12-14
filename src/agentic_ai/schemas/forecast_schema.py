from pydantic import BaseModel
from typing import List

class ForecastResult(BaseModel):
    metrics: dict
    explanations: List[str]
