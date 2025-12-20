# apps/backend/api/executive.py

from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd

from src.agentic_ai.narrative_agent import NarrativeAgent

router = APIRouter(tags=["Executive"])


class ExecutiveRequest(BaseModel):
    reorder: list
    horizon_days: int


@router.post("/summary")
def executive_summary(request: ExecutiveRequest):
    """
    COO-level executive narrative.
    """

    reorder_df = pd.DataFrame(request.reorder)

    agent = NarrativeAgent()  # safe fallback (no OpenAI key)

    summary = agent.generate_coo_summary(
        reorder_df=reorder_df,
        sim_df=None,
        forecast_horizon_days=request.horizon_days
    )

    return {
        "status": "success",
        "summary": summary
    }
