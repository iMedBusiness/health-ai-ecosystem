from src.ai_core.suppliers.supplier_repository import SupplierRepository
from src.ai_core.suppliers.supplier_ranker import SupplierRanker
from src.ai_core.inventory.lot_repository import LotRepository
from src.ai_core.inventory.expiry_risk import ExpiryRiskEngine
from src.ai_core.optimization.procurement_optimizer import ProcurementOptimizer
from src.ai_core.optimization.policy import normal_policy, emergency_policy

facility = "facility_a"
item = "amoxicillin_500mg"
required_qty = 30000

# Suppliers
srepo = SupplierRepository()
suppliers = srepo.get_suppliers(facility, item)
ranked = SupplierRanker().rank(suppliers)

# Expiry risk
lrepo = LotRepository()
lots = lrepo.get_lots(facility, item)
risk = ExpiryRiskEngine().compute(lots)
pct_at_risk_90 = risk.pct_at_risk_90

opt = ProcurementOptimizer()

# Normal
sol_n, meta_n = opt.optimize(ranked, required_qty, pct_at_risk_90, normal_policy())
print("\n=== NORMAL MODE ===")
print("META:", meta_n)
print(sol_n)

# Emergency
sol_e, meta_e = opt.optimize(ranked, required_qty, pct_at_risk_90, emergency_policy())
print("\n=== EMERGENCY MODE ===")
print("META:", meta_e)
print(sol_e)
