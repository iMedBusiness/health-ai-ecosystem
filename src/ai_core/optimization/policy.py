from src.ai_core.optimization.procurement_optimizer import OptimizationConfig


def normal_policy() -> OptimizationConfig:
    return OptimizationConfig(
        mode="normal",
        max_share_normal=0.70,
        max_share_emergency=0.50,
        shortage_penalty_per_unit=5.0,
        expiry_penalty_rate=0.25,
        weight_procurement=1.0,
        weight_expiry=1.0,
        weight_shortage=1.0
    )


def emergency_policy() -> OptimizationConfig:
    return OptimizationConfig(
        mode="emergency",
        max_share_normal=0.70,
        max_share_emergency=0.50,
        shortage_penalty_per_unit=50.0,   # much higher penalty
        expiry_penalty_rate=0.10,         # waste secondary in emergency
        weight_procurement=0.6,           # price matters less
        weight_expiry=0.3,
        weight_shortage=3.0              # shortage matters much more
    )
