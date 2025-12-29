from src.ai_core.suppliers.supplier_repository import SupplierRepository
from src.ai_core.suppliers.supplier_ranker import SupplierRanker
from src.ai_core.suppliers.allocation_engine import AllocationEngine

repo = SupplierRepository()
ranker = SupplierRanker()
allocator = AllocationEngine()

df = repo.get_suppliers(
    facility="facility_a",
    item="amoxicillin_500mg"
)

ranked = ranker.rank(df)

allocation = allocator.allocate(
    ranked_df=ranked,
    required_qty=30000,
    mode="normal"
)

print(allocation)
