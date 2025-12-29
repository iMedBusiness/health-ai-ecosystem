from src.ai_core.inventory.lot_repository import LotRepository
from src.ai_core.inventory.fefo_allocator import FEFOAllocator

repo = LotRepository()
allocator = FEFOAllocator()

lots = repo.get_lots(
    facility="facility_a",
    item="amoxicillin_500mg"
)

print("LOTS:")
print(lots)

fefo_result = allocator.allocate(
    lots_df=lots,
    required_qty=15000
)

print("\nFEFO ALLOCATION:")
print(fefo_result)
