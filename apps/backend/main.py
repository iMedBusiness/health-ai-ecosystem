from fastapi import FastAPI
from apps.backend.api.forecast import router as forecast_router
from apps.backend.api.reorder import router as reorder_router
from apps.backend.api.simulate import router as simulate_router
from apps.backend.api.executive import router as executive_router

app = FastAPI(
    title="Health AI Ecosystem API",
    version="0.1.0"
)

app.include_router(forecast_router, prefix="/forecast")
app.include_router(reorder_router, prefix="/reorder")
app.include_router(simulate_router, prefix="/simulate")
app.include_router(executive_router, prefix="/executive")

@app.get("/")
def root():
    return {"status": "Health AI Ecosystem API running"}
