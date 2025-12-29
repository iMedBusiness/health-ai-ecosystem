from src.ai_core.suppliers.shortage_sourcing import ShortageSourcingEngine, ShortageContext

engine = ShortageSourcingEngine()

ctx = ShortageContext(
    facility="facility_a",
    item="amoxicillin_500mg",
    days_of_cover=3.0,       # force shortage
    required_qty=30000
)

result = engine.emergency_plan(ctx)

print(result["summary"])
print("\nAllocation:")
print(result["allocation"])
