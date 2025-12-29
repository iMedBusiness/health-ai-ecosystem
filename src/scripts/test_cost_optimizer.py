from src.ai_core.suppliers.supplier_repository import SupplierRepository
from src.ai_core.suppliers.supplier_ranker import SupplierRanker
from src.ai_core.inventory.lot_repository import LotRepository
from src.ai_core.inventory.expiry_risk import ExpiryRiskEngine
from src.ai_core.optimization.procurement_optimizer import ProcurementOptimizer, OptimizationConfig

facility = "facility_a"
item = "amoxicillin_500mg"
required_qty = 30000

# Load suppliers
srepo = SupplierRepository()
suppliers = srepo.get_suppliers(facility, item)

# (Optional) rank first to filter top-N later; for now keep all
ranked = SupplierRanker().rank(suppliers)

# Load lots and compute expiry risk
lrepo = LotRepository()
lots = lrepo.get_lots(facility, item)
risk = ExpiryRiskEngine().compute(lots)
pct_at_risk_90 = risk.pct_at_risk_90

# Optimize
opt = ProcurementOptimizer()

config = OptimizationConfig(
    mode="normal",
    shortage_penalty_per_unit=5.0,
    expiry_penalty_rate=0.25
)

solution, meta = opt.optimize(
    suppliers_df=ranked,              # includes price/capacity/MOQ fields
    required_qty=required_qty,
    pct_at_risk_90=pct_at_risk_90,
    config=config
)

print("META:", meta)
print("\nSOLUTION:")
print(solution)
