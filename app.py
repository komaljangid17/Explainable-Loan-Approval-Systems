import streamlit as st
import numpy as np
import pickle
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import datetime
import matplotlib.pyplot as plt

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Loan AI Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model/model.pkl", "rb"))

# ---------------- PDF FUNCTION ----------------
def generate_pdf(prediction, prob, income, loan, cibil):

    doc = SimpleDocTemplate("loan_report.pdf")
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Loan Prediction Report", styles['Title']))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Date: {datetime.datetime.now()}", styles['Normal']))
    content.append(Paragraph(f"Prediction: {prediction}", styles['Normal']))
    content.append(Paragraph(f"Confidence: {round(prob*100,2)}%", styles['Normal']))
    content.append(Paragraph(f"Income: {income}", styles['Normal']))
    content.append(Paragraph(f"Loan Amount: {loan}", styles['Normal']))
    content.append(Paragraph(f"CIBIL Score: {cibil}", styles['Normal']))

    doc.build(content)

# ---------------- HEADER ----------------
st.title("💰 Smart Loan Approval AI System")
st.markdown("### AI-powered decision with risk analysis & insights")

tab1, tab2, tab3 = st.tabs(["🧾 Input Form", "📊 Result", "🧠 Insights"])

# ---------------- INPUT TAB ----------------
with tab1:
    st.subheader("Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        dependents = st.number_input("Dependents", 0, 10)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_emp = st.selectbox("Self Employed", ["Yes", "No"])
        income = st.number_input("Annual Income")

    with col2:
        loan_amount = st.number_input("Loan Amount")
        loan_term = st.number_input("Loan Term (months)")
        cibil = st.number_input("CIBIL Score")

    st.subheader("Assets Details")
    res_assets = st.number_input("Residential Assets")
    com_assets = st.number_input("Commercial Assets")
    lux_assets = st.number_input("Luxury Assets")
    bank_assets = st.number_input("Bank Assets")

    edu = 0 if education == "Graduate" else 1
    emp = 1 if self_emp == "Yes" else 0

    predict_btn = st.button("🚀 Predict Loan")

# ---------------- PREDICTION ----------------
if predict_btn:

    input_data = np.array([[
        0,
        dependents,
        edu,
        emp,
        income,
        loan_amount,
        loan_term,
        cibil,
        res_assets,
        com_assets,
        lux_assets,
        bank_assets
    ]])

    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    st.session_state["pred"] = pred[0]
    st.session_state["prob"] = prob
    st.session_state["input"] = {
        "income": income,
        "loan": loan_amount,
        "cibil": cibil
    }

# ---------------- RESULT TAB ----------------
with tab2:

    if "pred" in st.session_state:

        st.subheader("📊 Prediction Result")

        if st.session_state["pred"] == 1:
            st.success(f"✅ Loan Approved ({round(st.session_state['prob']*100,2)}%)")
            risk = "Low 🟢"
        else:
            st.error(f"❌ Loan Rejected ({round((1-st.session_state['prob'])*100,2)}%)")
            risk = "High 🔴"

        st.metric("Risk Level", risk)

        # ---------------- PDF DOWNLOAD ----------------
        if st.button("📥 Download Report"):

            pred_text = "Approved" if st.session_state["pred"] == 1 else "Rejected"

            generate_pdf(
                pred_text,
                st.session_state["prob"],
                st.session_state["input"]["income"],
                st.session_state["input"]["loan"],
                st.session_state["input"]["cibil"]
            )

            st.success("📄 PDF Generated Successfully!")

    else:
        st.info("Please make a prediction first")

# ---------------- INSIGHTS TAB ----------------
with tab3:

    if "input" in st.session_state:

        st.subheader("🧠 AI Insights")

        income = st.session_state["input"]["income"]
        loan = st.session_state["input"]["loan"]
        cibil = st.session_state["input"]["cibil"]

        reasons = []

        if cibil < 650:
            reasons.append("Low CIBIL Score")
        if income < 300000:
            reasons.append("Low Income")
        if loan > income:
            reasons.append("Loan > Income (High Risk)")

        if len(reasons) == 0:
            st.success("✔ Strong Financial Profile")
        else:
            for r in reasons:
                st.warning(r)

        # ---------------- GRAPH ----------------
        st.subheader("📉 Financial Comparison")

        chart_data = pd.DataFrame({
            "Category": ["Income", "Loan Amount"],
            "Value": [income, loan]
        })

        st.bar_chart(chart_data.set_index("Category"))

        # ---------------- EXTRA GRAPH ----------------
        st.subheader("📊 Risk Visualization")

        fig, ax = plt.subplots()
        ax.bar(["CIBIL Score"], [cibil])
        st.pyplot(fig)

    else:
        st.info("No insights yet")