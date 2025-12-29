from src.ai_core.inventory.lot_repository import LotRepository

repo = LotRepository()

lots = repo.get_lots(
    facility="facility_a",
    item="amoxicillin_500mg"
)

print(lots)
