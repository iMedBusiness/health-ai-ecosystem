from fastapi import APIRouter, HTTPException
import pandas as pd

from apps.backend.schemas.requests import ExecutiveSummaryRequest
from src.agentic_ai.narrative_agent import NarrativeAgent

router = APIRouter(tags=["Executive"])

@router.post("/summary")
def executive_summary(request: ExecutiveSummaryRequest):

    reorder_df = pd.DataFrame(request.reorder)
    vol_df = pd.DataFrame(request.volatility)
    risk_df = pd.DataFrame(request.inventory_risk)

    if reorder_df.empty:
        raise HTTPException(
            status_code=400,
            detail="Empty reorder payload received"
        )
    # -----------------------------
    # Merge volatility into reorder
    # -----------------------------
    if not vol_df.empty:
        for c in ["facility", "item"]:
            reorder_df[c] = reorder_df[c].astype(str).str.strip().str.lower()
            vol_df[c] = vol_df[c].astype(str).str.strip().str.lower()

        # Normalize volatility column name
        if "volatility_class" not in vol_df.columns and "volatility" in vol_df.columns:
            vol_df = vol_df.rename(columns={"volatility": "volatility_class"})

        reorder_df = reorder_df.merge(
            vol_df[["facility", "item", "volatility_class"]],
            on=["facility", "item"],
            how="left"
        )   

    # -----------------------------
    # Inventory risk (item-level)
    # -----------------------------
    if not risk_df.empty:
        for c in ["facility", "item"]:
            risk_df[c] = risk_df[c].astype(str).str.strip().str.lower()
        # Count risks
        high_risk = risk_df[risk_df["inventory_risk"] == "HIGH"]
        medium_risk = risk_df[risk_df["inventory_risk"] == "MEDIUM"]
        low_risk = risk_df[risk_df["inventory_risk"] == "LOW"]

        # Attach executive flags to reorder_df for narrative context
        reorder_df = reorder_df.merge(
            risk_df[["facility", "item", "inventory_risk"]],
            on=["facility", "item"],
            how="left"
        )

        reorder_df["executive_flag"] = reorder_df["inventory_risk"].fillna("UNKNOWN")


    # -----------------------------
    # Generate narrative
    # -----------------------------
    agent = NarrativeAgent()  # rule-based only
    summary = agent.generate_coo_summary(
        reorder_df=reorder_df,
        horizon_days=request.horizon_days
    )

    return {
        "status": "success",
        "summary": summary
    }
    

