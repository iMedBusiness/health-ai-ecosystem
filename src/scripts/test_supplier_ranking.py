from src.ai_core.suppliers.supplier_repository import SupplierRepository
from src.ai_core.suppliers.supplier_ranker import SupplierRanker

repo = SupplierRepository()
ranker = SupplierRanker()

df = repo.get_suppliers(
    facility="facility_a",
    item="amoxicillin_500mg"
)

ranked = ranker.rank(df)

print(
    ranked[
        [
            "supplier_id",
            "supplier_name",
            "supplier_score",
            "rank",
            "price_per_unit",
            "lead_time_days",
            "reliability_score",
            "risk_score"
        ]
    ]
)
