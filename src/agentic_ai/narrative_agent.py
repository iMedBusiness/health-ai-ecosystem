# src/agentic_ai/narrative_agent.py

import pandas as pd


class NarrativeAgent:
    def __init__(self):
        pass
    """
    Generates executive-level narrative insights.
    Safe by default (rule-based), LLM optional.
    """

    def generate_coo_summary(
        self,
        reorder_df: pd.DataFrame,
        sim_df = None,
        forecast_horizon_days = 30
    ) -> str:

        if reorder_df.empty:
            return "No reorder risks detected in the selected horizon."

        # Top risks
        top_risk = reorder_df.sort_values(
            "reorder_point", ascending=False
        ).head(5)

        lines = []
        lines.append("### 🧠 Executive Summary (COO)")
        lines.append("")
        lines.append(
            f"- Forecast horizon: **{horizon_days} days**"
        )
        lines.append(
            f"- **{len(reorder_df)}** item–facility combinations analyzed"
        )
        lines.append("")
        lines.append("#### 🔴 Highest Replenishment Risks")

        for _, r in top_risk.iterrows():
            lines.append(
                f"- **{r['item']}** @ **{r['facility']}** → "
                f"Reorder Point: **{round(r['reorder_point'], 1)}**, "
                f"Safety Stock: **{round(r['safety_stock'], 1)}**"
            )

        lines.append("")
        lines.append("#### ✅ Recommendations")
        lines.append("- Prioritize procurement for the above items")
        lines.append("- Review supplier lead-time reliability")
        lines.append("- Consider stock rebalancing across facilities")
        lines.append("- Monitor demand volatility weekly")

        return "\n".join(lines)


