import pandas as pd


class NarrativeAgent:
    """
    Generates COO-level executive summaries
    from forecast, reorder, and inventory simulation outputs.
    """

    def __init__(self, api_key=None):
        self.api_key = api_key

    def generate_coo_summary(
        self,
        reorder_df,
        sim_df,
        forecast_horizon_days
    ):
        # -------------------------------
        # DATA EXTRACTION
        # -------------------------------
        urgent_reorders = reorder_df.sort_values(
            "reorder_point", ascending=False
        ).head(3)

        stockout_risks = (
            sim_df[sim_df["stockout_date"].notna()]
            .sort_values("days_of_cover")
            .head(3)
        )

        facilities_at_risk = (
            sim_df.groupby("facility")
            .agg({"reorder_now": "sum"})
            .sort_values("reorder_now", ascending=False)
            .head(3)
        )

        if reorder_df is None or reorder_df.empty:
            return "⚠️ No reorder data available to generate executive summary."

        if sim_df is None or sim_df.empty:
            return "⚠️ No inventory simulation data available to generate executive summary."
        # -------------------------------
        # BASE NARRATIVE (RULE-BASED)
        # -------------------------------
        summary = f"""
### 📌 Executive Summary for COO

**Time horizon:** next {forecast_horizon_days} days

#### 🚨 Top Stockout Risks
"""

        if len(stockout_risks) == 0:
            summary += "- No stockouts projected within forecast horizon.\n"
        else:
            for _, r in stockout_risks.iterrows():
                summary += (
                    f"- **{r['item']}** at **{r['facility']}** "
                    f"(stockout in ~{r['days_of_cover']} days)\n"
                )

        summary += "\n#### 📦 Immediate Reorder Priorities\n"

        for _, r in urgent_reorders.iterrows():
            summary += (
                f"- **{r['item']}** at **{r['facility']}** "
                f"(ROP: {r['reorder_point']})\n"
            )

        summary += "\n#### 🏥 Facilities at Highest Risk\n"

        for facility, row in facilities_at_risk.iterrows():
            summary += (
                f"- **{facility}**: {int(row['reorder_now'])} items "
                f"require immediate reorder\n"
            )

        summary += """
#### ✅ Recommended Actions
- Prioritize procurement for high-risk items immediately
- Review supplier lead times for volatile items
- Rebalance inventory across facilities where possible
- Monitor daily demand variance for early warning signals
"""

        # -------------------------------
        # OPTIONAL LLM ENHANCEMENT
        # -------------------------------
        if self.api_key:
            try:
                import openai
                openai.api_key = self.api_key

                prompt = (
                    "Rewrite the following supply chain executive summary "
                    "for a COO in a concise, strategic tone:\n\n"
                    + summary
                )

                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )

                return response.choices[0].message["content"]

            except Exception:
                return summary

        return summary

