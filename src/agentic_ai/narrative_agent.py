import openai

class NarrativeAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def generate_summary(self, reorder_df):
        """
        Generate executive summary.
        Falls back to rule-based text if no API key is provided.
        """

        # -----------------------------
        # RULE-BASED FALLBACK (SAFE)
        # -----------------------------
        high_risk = reorder_df.sort_values(
            "reorder_point", ascending=False
        ).head(3)

        summary = (
            "### Executive Summary (COO)\n\n"
            f"- {len(reorder_df)} item–facility combinations analyzed\n"
            "- Top replenishment risks identified:\n"
        )

        for _, r in high_risk.iterrows():
            summary += (
                f"  • **{r['item']}** at **{r['facility']}** "
                f"(Reorder Point: {r['reorder_point']})\n"
            )

        summary += (
            "\n**Recommendations:**\n"
            "- Prioritize procurement for high-risk items\n"
            "- Review supplier lead-time stability\n"
            "- Consider redistributing stock across facilities\n"
        )

        # -----------------------------
        # OPTIONAL LLM ENHANCEMENT
        # -----------------------------
        if self.api_key:
            try:
                import openai
                openai.api_key = self.api_key

                prompt = (
                    "Rewrite the following executive summary in a more "
                    "concise, strategic tone for a COO:\n\n" + summary
                )

                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )

                return response.choices[0].message["content"]

            except Exception:
                # If OpenAI fails, fall back safely
                return summary

        return summary
