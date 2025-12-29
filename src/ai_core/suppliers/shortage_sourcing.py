from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd

from src.ai_core.suppliers.supplier_repository import SupplierRepository
from src.ai_core.suppliers.supplier_ranker import SupplierRanker
from src.ai_core.suppliers.allocation_engine import AllocationEngine
from src.ai_core.optimization.procurement_optimizer import ProcurementOptimizer
from src.ai_core.optimization.policy import emergency_policy

EMERGENCY_WEIGHTS = {
    "price": 0.10,
    "lead_time": 0.40,
    "reliability": 0.40,
    "risk": 0.10
}


@dataclass
class ShortageContext:
    facility: str
    item: str
    days_of_cover: float
    required_qty: float
    trigger_doc_days: float = 7.0
    service_level: Optional[float] = None
    service_level_threshold: float = 0.90


class ShortageSourcingEngine:
    """
    Detect shortage and output emergency sourcing plan:
    - ranks suppliers with emergency weights
    - allocates quantities using emergency diversification
    """

    def __init__(self):
        self.repo = SupplierRepository()
        self.allocator = AllocationEngine()

    def is_shortage(self, ctx: ShortageContext) -> (bool, str):
        if ctx.days_of_cover <= ctx.trigger_doc_days:
            return True, f"days_of_cover={ctx.days_of_cover:.2f} <= trigger={ctx.trigger_doc_days:.2f}"

        if ctx.service_level is not None and ctx.service_level < ctx.service_level_threshold:
            return True, f"service_level={ctx.service_level:.2f} < threshold={ctx.service_level_threshold:.2f}"

        return False, "no_shortage_trigger"

    def emergency_plan(self, ctx: ShortageContext) -> Dict[str, Any]:
        shortage, reason = self.is_shortage(ctx)

        suppliers = self.repo.get_suppliers(ctx.facility, ctx.item)

        if suppliers.empty:
            return {
                "shortage": shortage,
                "trigger_reason": reason,
                "error": "No suppliers found in supplier_pool for this facility-item",
                "allocation": None,
                "ranked_suppliers": None
            }

        # Expand pool: contracted first, then all
        contracted = suppliers[suppliers["contracted"] == True]
        pool = contracted if not contracted.empty else suppliers

        # Rank using emergency weights
        ranker = SupplierRanker(weights=EMERGENCY_WEIGHTS)
        ranked = ranker.rank(pool)

        # Optimize procurement using MILP (Phase 5)
        opt = ProcurementOptimizer()

        solution, meta = opt.optimize(
            suppliers_df=ranked,
            required_qty=ctx.required_qty,
            pct_at_risk_90=0.0,  # emergency: expiry risk downweighted
            config=emergency_policy()
        )

        # Simple explainability summary
        summary = {
            "shortage": shortage,
            "trigger_reason": reason,
            "mode": "emergency",
            "facility": ctx.facility,
            "item": ctx.item,
            "required_qty": ctx.required_qty,
            "optimizer_status": meta["status"],
            "residual_shortage": meta["shortage"],
            "objective_value": meta["objective_value"],
            "top_suppliers": ranked.head(3)[
                [
                    "supplier_id",
                    "supplier_name",
                    "supplier_score",
                    "price_per_unit",
                    "lead_time_days",
                    "reliability_score",
                    "risk_score",
                ]
            ].to_dict(orient="records"),
        }


        return {
            "summary": summary,
            "ranked_suppliers": ranked,
            "allocation": solution,        # optimized allocation
            "residual_shortage": meta["shortage"]
        } 
