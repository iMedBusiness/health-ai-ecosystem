from fastapi import APIRouter

router = APIRouter()

@router.post("/compute")
def compute_reorder(payload: dict):
    return {
        "status": "ok",
        "message": "Reorder point endpoint ready"
    }
