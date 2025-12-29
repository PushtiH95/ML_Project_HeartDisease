

# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
# import plotly.express as px
# import plotly.graph_objects as go
# from fpdf import FPDF

# # -------------------------------
# # Page Config & Styles
# # -------------------------------
# st.set_page_config(page_title="HeartGuard AI Pro", page_icon="💓", layout="wide")

# st.markdown("""
# <style>
#     .stApp { background: #0b0e14; color: #e0e0e0; }
#     .main-card {
#         background: rgba(23, 28, 40, 0.95);
#         border: 1px solid #30363d;
#         border-radius: 20px;
#         padding: 40px;
#         box-shadow: 0 10px 30px rgba(0,0,0,0.5);
#     }
#     .metric-box {
#         background: #1c2128;
#         border-radius: 15px;
#         padding: 20px;
#         border-top: 4px solid #4facfe;
#         text-align: center;
#     }
#     .suggestion-card {
#         background: rgba(79, 172, 254, 0.05);
#         border-left: 5px solid #4facfe;
#         padding: 20px;
#         border-radius: 10px;
#         margin-bottom: 15px;
#     }
# </style>
# """, unsafe_allow_html=True)

# # -------------------------------
# # Load Model & Scaler
# # -------------------------------
# @st.cache_resource
# def load_assets():
#     with open("heart_model_v2.pkl", "rb") as f:
#         model = pickle.load(f)
#     with open("scaler_v2.pkl", "rb") as f:
#         scaler = pickle.load(f)
#     return model, scaler

# model, scaler = load_assets()

# # Load dataset
# data = pd.read_csv("HeartD.csv")
# data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

# # -------------------------------
# # PDF Generator
# # -------------------------------
# def create_pdf(res):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_font("Arial", 'B', 16)
#     pdf.cell(200, 10, "HeartGuard AI Diagnostic Report", ln=True, align="C")
#     pdf.ln(10)
#     pdf.set_font("Arial", size=12)
#     pdf.cell(200, 10, f"Risk Probability: {res['prob']*100:.1f}%", ln=True)
#     pdf.cell(200, 10, f"Risk Status: {'HIGH' if res['pred']==1 else 'LOW'}", ln=True)
#     pdf.cell(200, 10, f"BMI: {res['bmi']:.2f}", ln=True)
#     pdf.ln(10)
#     pdf.multi_cell(0, 10, "Disclaimer: This AI report is for informational purposes only.")
#     return pdf.output(dest="S").encode("latin-1")

# # Session state
# if "screen" not in st.session_state:
#     st.session_state.screen = "input"

# # -------------------------------
# # SCREEN 1 – INPUT
# # -------------------------------
# if st.session_state.screen == "input":
#     st.markdown("<h1 style='text-align:center;color:#4facfe;'>HEARTGUARD NEURAL ENGINE</h1>", unsafe_allow_html=True)

#     with st.container():
#         st.markdown("<div class='main-card'>", unsafe_allow_html=True)
#         col1, col2 = st.columns(2)

#         with col1:
#             st.markdown("### 👤 Profile")
#             age = st.slider("Age", 18, 100, 45)
#             gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
#             height = st.slider("Height (cm)", 140, 200, 165)
#             weight = st.slider("Weight (kg)", 40, 150, 70)
#             chol = st.select_slider("Cholesterol", [1, 2, 3])

#         with col2:
#             st.markdown("### 💓 Vitals")
#             sbp = st.number_input("Systolic BP", 80, 200, 120)
#             dbp = st.number_input("Diastolic BP", 50, 130, 80)
#             gluc = st.select_slider("Glucose", [1, 2, 3])
#             smoke, alco, active = st.columns(3)
#             sm = smoke.toggle("Smoking")
#             al = alco.toggle("Alcohol")
#             ac = active.toggle("Active", value=True)

#         if st.button("RUN DIAGNOSTIC ANALYSIS ⚡"):
#             bmi = weight / ((height / 100) ** 2)
#             input_df = pd.DataFrame(
#                 [[age, gender, height, weight, sbp, dbp, chol, gluc, int(sm), int(al), int(ac), bmi]],
#                 columns=scaler.feature_names_in_
#             )
#             scaled = scaler.transform(input_df)
#             st.session_state.results = {
#                 "prob": model.predict_proba(scaled)[0][1],
#                 "pred": model.predict(scaled)[0],
#                 "bmi": bmi,
#                 "raw": input_df
#             }
#             st.session_state.screen = "result"
#             st.rerun()

#         st.markdown("</div>", unsafe_allow_html=True)

# # -------------------------------
# # SCREEN 2 – RESULTS
# # -------------------------------
# else:
#     res = st.session_state.results
#     raw = res["raw"].iloc[0]

#     st.markdown("<h2 style='text-align:center;'>NEURAL DIAGNOSTIC REPORT</h2>", unsafe_allow_html=True)

#     col1, col2 = st.columns(2)

#     # Gauge
#     with col1:
#         fig_gauge = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=res["prob"] * 100,
#             number={"suffix": "%"},
#             title={"text": "Risk Probability"},
#             gauge={"axis": {"range": [0, 100]}}
#         ))
#         st.plotly_chart(fig_gauge, use_container_width=True)

#     # -------------------------------
#     # FIXED RADAR CHART ✅
#     # -------------------------------
#     with col2:
#         st.markdown("### 🧬 Patient vs Dataset Average")

#         avg_vitals = data.mean(numeric_only=True)
#         categories = ["Age", "Systolic BP", "Diastolic BP", "BMI"]

#         fig_radar = go.Figure()

#         fig_radar.add_trace(go.Scatterpolar(
#             r=[raw["age"], raw["systolic_bp"], raw["diastolic_bp"], res["bmi"]],
#             theta=categories,
#             fill="toself",
#             name="Patient"
#         ))

#         fig_radar.add_trace(go.Scatterpolar(
#             r=[
#                 avg_vitals["age"],
#                 avg_vitals["systolic_bp"],
#                 avg_vitals["diastolic_bp"],
#                 avg_vitals["BMI"]
#             ],
#             theta=categories,
#             fill="toself",
#             name="Average"
#         ))

#         fig_radar.update_layout(
#             polar=dict(radialaxis=dict(visible=True, range=[0, 200])),
#             paper_bgcolor="rgba(0,0,0,0)",
#             font=dict(color="white")
#         )

#         st.plotly_chart(fig_radar, use_container_width=True)

#     # Correlation matrix
#     st.markdown("### 📊 Correlation Matrix")
#     corr = data.corr(numeric_only=True)
#     st.plotly_chart(px.imshow(corr, color_continuous_scale="RdBu_r"), use_container_width=True)

#     # Suggestions
#     st.markdown("### 🩺 Personalized Care Plan")
#     if res["prob"] > 0.5:
#         st.markdown("<div class='suggestion-card'><b>🚨 High Risk:</b> Consult a cardiologist.</div>", unsafe_allow_html=True)

#     # PDF Export
#     pdf_data = create_pdf(res)
#     st.download_button("📥 Download PDF Report", pdf_data, "Heart_Report.pdf", "application/pdf")

#     if st.button("🔁 RESCAN"):
#         st.session_state.screen = "input"
#         st.rerun()


import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF

# ===============================
# PAGE CONFIG (MOBILE FRIENDLY)
# ===============================
st.set_page_config(
    page_title="HeartGuard Neo",
    page_icon="💓",
    layout="wide"
)

# ===============================
# UNIQUE MEDICAL DARK THEME
# ===============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b0e14, #111827);
    color: #e5e7eb;
}
.card {
    background: rgba(17, 24, 39, 0.95);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 0 25px rgba(79,172,254,0.15);
    margin-bottom: 20px;
}
.badge-high {
    background: #7f1d1d;
    color: #fecaca;
    padding: 12px;
    border-radius: 12px;
    text-align: center;
}
.badge-low {
    background: #064e3b;
    color: #bbf7d0;
    padding: 12px;
    border-radius: 12px;
    text-align: center;
}
.suggestion {
    background: rgba(79,172,254,0.08);
    border-left: 5px solid #4facfe;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.footer {
    text-align: center;
    color: #9ca3af;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL & SCALER
# ===============================
@st.cache_resource
def load_assets():
    with open("heart_model_v2.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler_v2.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# ===============================
# LOAD DATASET
# ===============================
data = pd.read_csv("HeartD.csv")
data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

# ===============================
# PDF REPORT
# ===============================
def generate_pdf(res, explanation):
    pdf = FPDF()
    pdf.add_page()

    # Unicode fonts
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf", uni=True)

    pdf.set_font("DejaVu", "B", 16)
    pdf.cell(0, 10, "HeartGuard Neo – Diagnostic Report", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("DejaVu", size=12)
    pdf.cell(0, 8, f"Risk Probability: {res['prob']*100:.1f}%", ln=True)
    pdf.cell(0, 8, f"Risk Status: {'HIGH' if res['pred']==1 else 'LOW'}", ln=True)
    pdf.cell(0, 8, f"BMI: {res['bmi']:.2f}", ln=True)
    pdf.ln(6)

    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 8, "Why this result?", ln=True)

    pdf.set_font("DejaVu", size=11)
    pdf.multi_cell(0, 7, explanation)

    pdf.ln(4)
    pdf.multi_cell(
        0, 7,
        "Disclaimer: This AI-generated report is for educational purposes only "
        "and must not replace professional medical consultation."
    )

    return pdf.output(dest="S").encode("utf-8")
# ===============================
# SESSION STATE
# ===============================
if "screen" not in st.session_state:
    st.session_state.screen = "input"

# ===============================
# SCREEN 1: INPUT
# ===============================
if st.session_state.screen == "input":

    st.markdown("<h1 style='text-align:center'>💓 HeartGuard Neo</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#9ca3af'>AI-powered cardiovascular risk assessment</p>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 18, 100, 45)
            gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
            height = st.slider("Height (cm)", 140, 200, 165)
            weight = st.slider("Weight (kg)", 40, 150, 70)

        with col2:
            systolic = st.number_input("Systolic BP", 80, 200, 120)
            diastolic = st.number_input("Diastolic BP", 50, 130, 80)
            cholesterol = st.select_slider("Cholesterol Level", [1, 2, 3])
            glucose = st.select_slider("Glucose Level", [1, 2, 3])
            active = st.checkbox("Physically Active", value=True)

        if st.button("🧠 Analyze Heart Risk"):
            with st.spinner("Analyzing health data..."):
                bmi = weight / ((height / 100) ** 2)

                input_df = pd.DataFrame([[
                    age, gender, height, weight,
                    systolic, diastolic,
                    cholesterol, glucose,
                    0, 0, int(active), bmi
                ]], columns=scaler.feature_names_in_)

                scaled = scaler.transform(input_df)

                st.session_state.results = {
                    "prob": model.predict_proba(scaled)[0][1],
                    "pred": model.predict(scaled)[0],
                    "bmi": bmi,
                    "raw": input_df.iloc[0]
                }

                st.session_state.screen = "result"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# SCREEN 2: RESULTS
# ===============================
else:
    res = st.session_state.results
    raw = res["raw"]

    st.markdown("<h2 style='text-align:center'>🧾 Diagnostic Report</h2>", unsafe_allow_html=True)

    # Risk Badge
    if res["pred"] == 1:
        st.markdown("<div class='badge-high'>🚨 HIGH CARDIOVASCULAR RISK</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='badge-low'>✅ LOW CARDIOVASCULAR RISK</div>", unsafe_allow_html=True)

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=res["prob"] * 100,
        number={"suffix": "%"},
        title={"text": "Risk Probability"},
        gauge={"axis": {"range": [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Explainability
    explanation = (
        f"The AI model evaluated your age ({raw['age']}), blood pressure "
        f"({raw['systolic_bp']}/{raw['diastolic_bp']} mmHg), BMI ({res['bmi']:.1f}) "
        "and lifestyle indicators. Elevated blood pressure and BMI increase "
        "cardiovascular strain, while physical activity reduces risk."
    )

    st.info("🧠 **Why this result?**\n\n" + explanation)

    # Suggestions
    st.markdown("### 🩺 What should you do?")
    if res["pred"] == 1:
        st.markdown("""
        <div class="suggestion">
        • Consult a cardiologist within 1–2 weeks  
        • Monitor BP daily  
        • Reduce salt & processed foods  
        • Avoid smoking & alcohol  
        • Begin doctor-approved physical activity
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="suggestion">
        • Continue regular exercise  
        • Maintain healthy diet  
        • Annual heart check-up  
        • Keep BP under 130/85  
        • Maintain ideal BMI
        </div>
        """, unsafe_allow_html=True)

    # Correlation Matrix
    st.markdown("### 📊 Dataset Insights")
    corr = data.corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, color_continuous_scale="RdBu_r"), use_container_width=True)

    # PDF Export
    pdf_bytes = generate_pdf(res, explanation)
    st.download_button("📄 Download Medical PDF Report", pdf_bytes, "HeartGuard_Report.pdf", "application/pdf")

    if st.button("🔁 New Assessment"):
        st.session_state.screen = "input"
        st.rerun()

# ===============================
# FOOTER
# ===============================
st.markdown("""
<div class="footer">
💓 <b>HeartGuard Neo</b> – AI for preventive cardiology  
<br>Educational use only
</div>
""", unsafe_allow_html=True)
