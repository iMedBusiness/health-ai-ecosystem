import pandas as pd
from dataclasses import dataclass


@dataclass
class ExpiryRiskResult:
    total_qty: float
    expiring_30: float
    expiring_60: float
    expiring_90: float
    pct_at_risk_90: float
    risk_class: str


class ExpiryRiskEngine:
    """
    Computes expiry risk metrics for lot-level inventory.
    """

    def compute(self, lots_df: pd.DataFrame, today=None) -> ExpiryRiskResult:
        df = lots_df.copy()

        if df.empty:
            return ExpiryRiskResult(
                total_qty=0.0,
                expiring_30=0.0,
                expiring_60=0.0,
                expiring_90=0.0,
                pct_at_risk_90=0.0,
                risk_class="LOW"
            )

        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
        df["qty_on_hand"] = df["qty_on_hand"].astype(float)

        if today is None:
            today = pd.Timestamp.today().normalize()
        else:
            today = pd.to_datetime(today).normalize()

        total_qty = float(df["qty_on_hand"].sum())

        def expiring_within(days: int) -> float:
            cutoff = today + pd.Timedelta(days=days)
            return float(
                df.loc[df["expiry_date"] <= cutoff, "qty_on_hand"].sum()
            )

        exp_30 = expiring_within(30)
        exp_60 = expiring_within(60)
        exp_90 = expiring_within(90)

        pct_at_risk_90 = exp_90 / total_qty if total_qty > 0 else 0.0

        # Risk classification (policy-driven, adjustable)
        if pct_at_risk_90 >= 0.30:
            risk_class = "HIGH"
        elif pct_at_risk_90 >= 0.10:
            risk_class = "MED"
        else:
            risk_class = "LOW"

        return ExpiryRiskResult(
            total_qty=round(total_qty, 2),
            expiring_30=round(exp_30, 2),
            expiring_60=round(exp_60, 2),
            expiring_90=round(exp_90, 2),
            pct_at_risk_90=round(pct_at_risk_90, 4),
            risk_class=risk_class
        )
