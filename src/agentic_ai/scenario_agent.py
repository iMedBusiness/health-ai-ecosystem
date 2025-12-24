import pandas as pd

class ScenarioAgent:
    """
    Runs what-if scenarios by perturbing forecast or lead time
    """

    def run_demand_surge(
        self,
        forecast_df: pd.DataFrame,
        surge_pct: float
    ) -> pd.DataFrame:
        df = forecast_df.copy()
        df["forecast"] = df["forecast"] * (1 + surge_pct)
        df["scenario"] = f"Demand +{int(surge_pct*100)}%"
        return df

    def run_lead_time_shock(
        self,
        forecast_df: pd.DataFrame,
        extra_days: int
    ) -> pd.DataFrame:
        df = forecast_df.copy()
        df["lead_time_days"] = df["lead_time_days"] + extra_days
        df["scenario"] = f"Lead Time +{extra_days}d"
        return df
