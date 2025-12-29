import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="Health AI — Decision Cockpit",
    layout="wide"
)

st.title("🧠 Health Supply Chain — Decision Cockpit")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Decision Context")

facility = st.sidebar.text_input("Facility", "facility_a")
item = st.sidebar.text_input("Item", "amoxicillin_500mg")
required_qty = st.sidebar.number_input(
    "Required Quantity",
    min_value=0,
    value=30000,
    step=1000
)
days_of_cover = st.sidebar.number_input(
    "Days of Cover",
    min_value=0.0,
    value=3.0,
    step=0.5
)

mode = st.sidebar.selectbox(
    "Decision Mode",
    options=["auto", "normal", "emergency"]
)

run_decision = st.sidebar.button("Run Decision")

# -----------------------------
# Main panel
# -----------------------------
if run_decision:

    with st.spinner("Running optimization..."):
        # 1️⃣ Call Decision API
        decision_payload = {
            "facility": facility,
            "item": item,
            "required_qty": required_qty,
            "days_of_cover": days_of_cover,
            "mode": mode
        }

        decision_resp = requests.post(
            f"{API_BASE}/decision/procurement",
            json=decision_payload
        )

        if decision_resp.status_code != 200:
            st.error("Decision API failed")
            st.stop()

        decision = decision_resp.json()

    # -----------------------------
    # Decision results
    # -----------------------------
    st.subheader("📦 Procurement Decision")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Decision Mode", decision["decision_mode"])
    col2.metric("Residual Shortage", f"{decision['residual_shortage']:,.0f}")
    col3.metric("Expected Cost", f"{decision['expected_cost']:,.2f}")
    col4.metric("Expiry Risk", decision["expiry_risk_class"])

    st.markdown("### Supplier Allocation")
    st.table(decision["procurement_plan"])

    # -----------------------------
    # Executive summary
    # -----------------------------
    with st.spinner("Generating executive summary..."):

        exec_payload = {
            "facility": facility,
            "item": item,
            "decision_mode": decision["decision_mode"],
            "trigger_reason": decision["trigger_reason"],
            "required_qty": required_qty,
            "residual_shortage": decision["residual_shortage"],
            "expected_cost": decision["expected_cost"],
            "expiry_risk_class": decision["expiry_risk_class"],
        }

        exec_resp = requests.post(
            f"{API_BASE}/executive/decision/executive-summary",
            json=exec_payload
        )

        if exec_resp.status_code != 200:
            st.error("Executive Summary API failed")
            st.stop()

        summary = exec_resp.json()

    st.subheader("🧾 Executive Summary")

    st.markdown(f"**Headline:** {summary['summary']['headline']}")
    st.markdown(f"**Situation:** {summary['summary']['situation']}")
    st.markdown(f"**Risk:** {summary['summary']['risk']}")
    st.markdown(f"**Cost Impact:** {summary['summary']['cost']}")
    st.markdown(f"**Gap:** {summary['summary']['gap']}")
