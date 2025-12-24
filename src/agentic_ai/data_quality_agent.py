# src/agentic_ai/data_quality_agent.py

import pandas as pd

class DataQualityAgent:
    def assess(self, df: pd.DataFrame) -> dict:
        issues = []
        score = 100

        if df.empty:
            return {
                "score": 0,
                "issues": ["No data provided"],
                "history_days": 0
            }

        history_days = df["ds"].nunique()

        if history_days < 30:
            issues.append("Insufficient historical data (<30 days)")
            score -= 30

        if df["y"].isna().mean() > 0.05:
            issues.append("Missing demand values detected")
            score -= 15

        if (df["y"] <= 0).any():
            issues.append("Zero or negative demand values detected")
            score -= 10

        return {
            "score": max(score, 0),
            "issues": issues,
            "history_days": history_days
        }
