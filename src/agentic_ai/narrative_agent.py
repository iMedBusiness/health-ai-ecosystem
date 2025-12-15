import openai

class NarrativeAgent:
    """
    Generates executive summaries / insights using OpenAI GPT.
    """
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_summary(self, forecast_df):
        """
        Returns a text summary highlighting top changes, risks, and recommendations.
        """
        # Simple example: summarize total forecast change by facility
        summary_prompt = f"""
        Generate a short executive summary for the COO based on the following forecast data:

        {forecast_df.groupby('facility')['forecast'].sum().to_dict()}

        Highlight:
        1. Facilities with highest forecast increase
        2. Potential stockout risks
        3. Recommendations
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.5
        )

        return response.choices[0].message["content"]
