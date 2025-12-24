# src/agentic_ai/confidence_agent.py

class ConfidenceAgent:
    def score(
        self,
        data_quality: dict,
        volatility_class: str,
        lead_time_missing: bool,
        forecast_cache_hit: bool
    ) -> dict:

        drivers = []
        warnings = []

        # --- Data Quality (30)
        dq_score = data_quality["score"] * 0.30
        if data_quality["issues"]:
            warnings.extend(data_quality["issues"])
        else:
            drivers.append("High data completeness")

        # --- Demand Stability (25)
        vol_map = {
            "Low": 25,
            "Medium": 15,
            "High": 5,
            "Unknown": 10
        }
        vol_score = vol_map.get(volatility_class, 10)
        if vol_score >= 20:
            drivers.append("Stable demand pattern")
        else:
            warnings.append("Demand volatility reduces confidence")

        # --- Forecast Reliability (25)
        fc_score = 25 if forecast_cache_hit else 15
        drivers.append(
            "Forecast model reused (cached)"
            if forecast_cache_hit else
            "Forecast model freshly trained"
        )

        # --- Lead Time Confidence (20)
        if lead_time_missing:
            lt_score = 10
            warnings.append("Lead time inferred (not observed)")
        else:
            lt_score = 20
            drivers.append("Observed lead time available")

        total = round(dq_score + vol_score + fc_score + lt_score)

        band = (
            "HIGH" if total >= 75 else
            "MEDIUM" if total >= 50 else
            "LOW"
        )

        return {
            "confidence_score": total,
            "confidence_band": band,
            "confidence_drivers": drivers,
            "confidence_warnings": warnings
        }
