import pandas as pd

class NarrativeAgent:
    """
    Rule-based executive narrative generator (COO-level).
    Volatility- and inventory-risk aware.
    """

    def generate_coo_summary(
        self,
        reorder_df,
        horizon_days: int
    ) -> str:

        total = len(reorder_df)

        # -----------------------------
        # Inventory risk distribution
        # -----------------------------
        risk_counts = (
            reorder_df["inventory_risk"]
            .value_counts(dropna=True)
            .to_dict()
            if "inventory_risk" in reorder_df.columns
            else {}
        )

        high_risk = risk_counts.get("HIGH", 0)
        med_risk = risk_counts.get("MEDIUM", 0)
        low_risk = risk_counts.get("LOW", 0)

        # -----------------------------
        # Volatility distribution
        # -----------------------------
        vol_counts = (
            reorder_df["volatility_class"]
            .value_counts(dropna=True)
            .to_dict()
            if "volatility_class" in reorder_df.columns
            else {}
        )

        erratic = vol_counts.get("Erratic", 0)
        seasonal = vol_counts.get("Seasonal", 0)
        stable = vol_counts.get("Stable", 0)

        # -----------------------------
        # High-risk drivers
        # -----------------------------
        high_risk_df = reorder_df[
            reorder_df["inventory_risk"] == "HIGH"
        ] if "inventory_risk" in reorder_df.columns else reorder_df.iloc[0:0]

        erratic_high = (
            high_risk_df["volatility_class"].eq("Erratic").sum()
            if "volatility_class" in high_risk_df.columns
            else 0
        )

        urgent_reorders = (
            high_risk_df["reorder_now"].sum()
            if "reorder_now" in high_risk_df.columns
            else 0
        )

        # -----------------------------
        # Top exposure items
        # -----------------------------
        if "days_of_cover" in reorder_df.columns:
            top_risk = reorder_df.sort_values(
                "days_of_cover", ascending=True
            ).head(5)
        else:
            # fallback to reorder pressure
            top_risk = reorder_df.sort_values(
                "reorder_point", ascending=False
            ).head(5)   
        # -----------------------------
        # Narrative
        # -----------------------------
        summary = f"""
### 🧠 Executive Summary — COO View

**Planning Horizon:** {horizon_days} days  
**Item–Facility Combinations Analyzed:** {total}

---

### 🚦 Inventory Risk Snapshot
- 🔴 **High Risk:** {high_risk}
- 🟠 **Medium Risk:** {med_risk}
- 🟢 **Low Risk:** {low_risk}

High-risk items represent **immediate stockout exposure** and require short-term operational intervention.

---

### 📈 Demand Volatility Overview
- **{erratic} combinations exhibit erratic demand** (highest planning uncertainty)
- **{seasonal} combinations show seasonal patterns**
- **{stable} combinations remain demand-stable**

Erratic demand is a **key driver of safety stock pressure and reorder volatility**.

---

### ⚠️ Key Risk Drivers Identified
- **{erratic_high} high-risk items are driven by erratic demand**
- **{urgent_reorders} high-risk items require immediate replenishment**
- Low days of cover combined with demand volatility is the **dominant stockout risk pattern**

---

### 🚨 Most Exposed Items (Lowest Days of Cover)
"""

        for _, r in top_risk.iterrows():
            summary += (
                f"- **{r['item']}** at **{r['facility']}** | "
                f"{'Days of Cover: ' + str(round(r['days_of_cover'], 1)) if 'days_of_cover' in r else 'ROP: ' + str(r['reorder_point'])} | "
                f"Risk: {r.get('inventory_risk', 'Unknown')} | "
                f"Volatility: {r.get('volatility_class', 'Unknown')}\n"
            )

        summary += """
---

### ✅ Executive Recommendations
**Immediate (Next 7–14 Days):**
- Expedite replenishment for **HIGH-risk items**
- Prioritize SKUs with **erratic demand and <7 days of cover**
- Validate supplier lead-time reliability for urgent reorders

**Structural (Next 30–90 Days):**
- Revisit service levels for erratic-demand SKUs
- Segment inventory policies by volatility class
- Consider inventory pooling or demand smoothing for unstable items
"""

        return summary.strip()
    

    def generate_decision_summary(self, decision_df):
        r = decision_df.iloc[0]

        if r["decision_mode"] == "emergency":
            headline = f"Emergency procurement required for {r['item']} at {r['facility']}."
        else:
            headline = f"Procurement optimization completed for {r['item']} at {r['facility']}."

        situation = f"Trigger condition: {r['trigger_reason']}."
        cost = f"Estimated procurement cost: {r['expected_cost']:,.2f}."
        risk = f"Expiry risk classified as {r['inventory_risk']}."

        if r["residual_shortage"] > 0:
            gap = (
                f"A residual shortage of {r['residual_shortage']:,.0f} units remains "
                "and requires executive escalation."
            )
        else:
            gap = "No residual supply gap is expected."

        return {
            "headline": headline,
            "situation": situation,
            "risk": risk,
            "cost": cost,
            "gap": gap,
        }

