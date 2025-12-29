from src.ai_core.inventory.lot_repository import LotRepository
from src.ai_core.inventory.expiry_risk import ExpiryRiskEngine

repo = LotRepository()
engine = ExpiryRiskEngine()

lots = repo.get_lots(
    facility="facility_a",
    item="amoxicillin_500mg"
)

print("LOTS:")
print(lots)

risk = engine.compute(lots)

print("\nEXPIRY RISK RESULT:")
print(risk)
