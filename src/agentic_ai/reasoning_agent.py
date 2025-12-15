import pandas as pd

class ReasoningAgent:
    """
    Lightweight explainability agent for demand changes
    """

    def explain_demand_change(self, df, forecast_df):
        explanations = []

        avg_stock = df["stock_on_hand"].mean()
        avg_lead = df["lead_time_days"].mean()
        demand_trend = forecast_df["forecast"].diff().mean()

        if demand_trend > 0:
            explanations.append("📈 Forecast shows increasing demand trend")

        if avg_stock < df["stock_on_hand"].quantile(0.25):
            explanations.append("⚠️ Low stock levels may drive replenishment demand")

        if avg_lead > df["lead_time_days"].quantile(0.75):
            explanations.append("⏳ Longer lead times increasing safety demand")

        if not explanations:
            explanations.append("📊 Demand driven by historical seasonality patterns")

        return explanations