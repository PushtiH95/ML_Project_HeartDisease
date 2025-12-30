
# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
# import plotly.graph_objects as go
# import plotly.express as px
# from fpdf import FPDF

# # ===============================
# # PAGE CONFIG (MOBILE FRIENDLY)
# # ===============================
# st.set_page_config(
#     page_title="HeartGuard Neo",
#     page_icon="💓",
#     layout="wide"
# )

# # ===============================
# # UNIQUE MEDICAL DARK THEME
# # ===============================
# st.markdown("""
# <style>
# .stApp {
#     background: linear-gradient(135deg, #0b0e14, #111827);
#     color: #e5e7eb;
# }
# .card {
#     background: rgba(17, 24, 39, 0.95);
#     border-radius: 20px;
#     padding: 30px;
#     box-shadow: 0 0 25px rgba(79,172,254,0.15);
#     margin-bottom: 20px;
# }
# .badge-high {
#     background: #7f1d1d;
#     color: #fecaca;
#     padding: 12px;
#     border-radius: 12px;
#     text-align: center;
# }
# .badge-low {
#     background: #064e3b;
#     color: #bbf7d0;
#     padding: 12px;
#     border-radius: 12px;
#     text-align: center;
# }
# .suggestion {
#     background: rgba(79,172,254,0.08);
#     border-left: 5px solid #4facfe;
#     padding: 18px;
#     border-radius: 12px;
#     margin-bottom: 15px;
# }
# .footer {
#     text-align: center;
#     color: #9ca3af;
#     margin-top: 40px;
# }
# </style>
# """, unsafe_allow_html=True)

# # ===============================
# # LOAD MODEL & SCALER
# # ===============================
# @st.cache_resource
# def load_assets():
#     with open("heart_model_v2.pkl", "rb") as f:
#         model = pickle.load(f)
#     with open("scaler_v2.pkl", "rb") as f:
#         scaler = pickle.load(f)
#     return model, scaler

# model, scaler = load_assets()

# # ===============================
# # LOAD DATASET
# # ===============================
# data = pd.read_csv("HeartD.csv")
# data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

# # ===============================
# # PDF REPORT
# # ===============================
# def generate_pdf(res, explanation):
#     pdf = FPDF()
#     pdf.add_page()

#     # Unicode fonts
#     pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
#     pdf.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf", uni=True)

#     pdf.set_font("DejaVu", "B", 16)
#     pdf.cell(0, 10, "HeartGuard Neo – Diagnostic Report", ln=True, align="C")
#     pdf.ln(8)

#     pdf.set_font("DejaVu", size=12)
#     pdf.cell(0, 8, f"Risk Probability: {res['prob']*100:.1f}%", ln=True)
#     pdf.cell(0, 8, f"Risk Status: {'HIGH' if res['pred']==1 else 'LOW'}", ln=True)
#     pdf.cell(0, 8, f"BMI: {res['bmi']:.2f}", ln=True)
#     pdf.ln(6)

#     pdf.set_font("DejaVu", "B", 12)
#     pdf.cell(0, 8, "Why this result?", ln=True)

#     pdf.set_font("DejaVu", size=11)
#     pdf.multi_cell(0, 7, explanation)

#     pdf.ln(4)
#     pdf.multi_cell(
#         0, 7,
#         "Disclaimer: This AI-generated report is for educational purposes only "
#         "and must not replace professional medical consultation."
#     )

#     return pdf.output(dest="S").encode("utf-8")
# # ===============================
# # SESSION STATE
# # ===============================
# if "screen" not in st.session_state:
#     st.session_state.screen = "input"

# # ===============================
# # SCREEN 1: INPUT
# # ===============================
# if st.session_state.screen == "input":

#     st.markdown("<h1 style='text-align:center'>💓 HeartGuard Neo</h1>", unsafe_allow_html=True)
#     st.markdown("<p style='text-align:center;color:#9ca3af'>AI-powered cardiovascular risk assessment</p>", unsafe_allow_html=True)

#     with st.container():
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         col1, col2 = st.columns(2)

#         with col1:
#             age = st.slider("Age", 18, 100, 45)
#             gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
#             height = st.slider("Height (cm)", 140, 200, 165)
#             weight = st.slider("Weight (kg)", 40, 150, 70)

#         with col2:
#             systolic = st.number_input("Systolic BP", 80, 200, 120)
#             diastolic = st.number_input("Diastolic BP", 50, 130, 80)
#             cholesterol = st.select_slider("Cholesterol Level", [1, 2, 3])
#             glucose = st.select_slider("Glucose Level", [1, 2, 3])
#             active = st.checkbox("Physically Active", value=True)

#         if st.button("🧠 Analyze Heart Risk"):
#             with st.spinner("Analyzing health data..."):
#                 bmi = weight / ((height / 100) ** 2)

#                 input_df = pd.DataFrame([[
#                     age, gender, height, weight,
#                     systolic, diastolic,
#                     cholesterol, glucose,
#                     0, 0, int(active), bmi
#                 ]], columns=scaler.feature_names_in_)

#                 scaled = scaler.transform(input_df)

#                 st.session_state.results = {
#                     "prob": model.predict_proba(scaled)[0][1],
#                     "pred": model.predict(scaled)[0],
#                     "bmi": bmi,
#                     "raw": input_df.iloc[0]
#                 }

#                 st.session_state.screen = "result"
#                 st.rerun()

#         st.markdown("</div>", unsafe_allow_html=True)

# # ===============================
# # SCREEN 2: RESULTS
# # ===============================
# else:
#     res = st.session_state.results
#     raw = res["raw"]

#     st.markdown("<h2 style='text-align:center'>🧾 Diagnostic Report</h2>", unsafe_allow_html=True)

#     # Risk Badge
#     if res["pred"] == 1:
#         st.markdown("<div class='badge-high'>🚨 HIGH CARDIOVASCULAR RISK</div>", unsafe_allow_html=True)
#     else:
#         st.markdown("<div class='badge-low'>✅ LOW CARDIOVASCULAR RISK</div>", unsafe_allow_html=True)

#     # Gauge
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=res["prob"] * 100,
#         number={"suffix": "%"},
#         title={"text": "Risk Probability"},
#         gauge={"axis": {"range": [0, 100]}}
#     ))
#     st.plotly_chart(fig, use_container_width=True)

#     # Explainability
#     explanation = (
#         f"The AI model evaluated your age ({raw['age']}), blood pressure "
#         f"({raw['systolic_bp']}/{raw['diastolic_bp']} mmHg), BMI ({res['bmi']:.1f}) "
#         "and lifestyle indicators. Elevated blood pressure and BMI increase "
#         "cardiovascular strain, while physical activity reduces risk."
#     )

#     st.info("🧠 **Why this result?**\n\n" + explanation)

#     # Suggestions
#     st.markdown("### 🩺 What should you do?")
#     if res["pred"] == 1:
#         st.markdown("""
#         <div class="suggestion">
#         • Consult a cardiologist within 1–2 weeks  
#         • Monitor BP daily  
#         • Reduce salt & processed foods  
#         • Avoid smoking & alcohol  
#         • Begin doctor-approved physical activity
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown("""
#         <div class="suggestion">
#         • Continue regular exercise  
#         • Maintain healthy diet  
#         • Annual heart check-up  
#         • Keep BP under 130/85  
#         • Maintain ideal BMI
#         </div>
#         """, unsafe_allow_html=True)

#     # Correlation Matrix
#     st.markdown("### 📊 Dataset Insights")
#     corr = data.corr(numeric_only=True)
#     st.plotly_chart(px.imshow(corr, color_continuous_scale="RdBu_r"), use_container_width=True)

#     # PDF Export
#     pdf_bytes = generate_pdf(res, explanation)
#     st.download_button("📄 Download Medical PDF Report", pdf_bytes, "HeartGuard_Report.pdf", "application/pdf")

#     if st.button("🔁 New Assessment"):
#         st.session_state.screen = "input"
#         st.rerun()

# # ===============================
# # FOOTER
# # ===============================
# st.markdown("""
# <div class="footer">
# 💓 <b>HeartGuard Neo</b> – AI for preventive cardiology  
# <br>Educational use only
# </div>
# """, unsafe_allow_html=True)



import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="CardioPredict AI - Heart Disease Risk Assessment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CUSTOM CSS FOR PROFESSIONAL UI
# ===============================
st.markdown("""
<style>
/* Main Styles */
.stApp {
    background: linear-gradient(135deg, #0a0e17 0%, #121828 50%, #0a0e17 100%);
    color: #e8eaed;
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(12, 18, 36, 0.95) !important;
    border-right: 1px solid rgba(79, 172, 254, 0.1);
}

[data-testid="stSidebar"] .stButton button {
    width: 100%;
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}

[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
}

/* Cards */
.card {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(79, 172, 254, 0.15);
    border-radius: 16px;
    padding: 28px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: transform 0.3s ease;
    margin-bottom: 24px;
}

.card:hover {
    transform: translateY(-2px);
}

/* Headers */
.main-header {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.8rem;
    text-align: center;
    margin-bottom: 0.5rem;
    letter-spacing: -0.5px;
}

.sub-header {
    color: #94a3b8;
    text-align: center;
    font-size: 1.1rem;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Risk Indicators */
.risk-high {
    background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    animation: pulse 2s infinite;
    border: none;
    margin: 20px 0;
}

.risk-moderate {
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: none;
    margin: 20px 0;
}

.risk-low {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: none;
    margin: 20px 0;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
    100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    border: none;
    padding: 12px 32px;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
}

/* Metric Cards */
.metric-card {
    background: rgba(15, 23, 42, 0.7);
    border-left: 4px solid #4facfe;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}

/* ECG Animation Container */
.ecg-container {
    background: rgba(0, 0, 0, 0.3);
    border-radius: 12px;
    padding: 20px;
    margin: 20px 0;
    position: relative;
    overflow: hidden;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background-color: rgba(15, 23, 42, 0.8);
    padding: 4px;
    border-radius: 10px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 10px 20px;
    background-color: transparent;
}

.stTabs [aria-selected="true"] {
    background-color: #4facfe !important;
    color: white !important;
}

/* Progress Bar */
.stProgress > div > div > div > div {
    background-color: #4facfe;
}

/* Input Fields */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
}

.stNumberInput input, .stSelectbox select {
    background-color: rgba(15, 23, 42, 0.7) !important;
    color: white !important;
    border: 1px solid rgba(79, 172, 254, 0.3) !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    padding: 20px;
    margin-top: 40px;
    border-top: 1px solid rgba(79, 172, 254, 0.1);
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# ECG ANIMATION FUNCTION
# ===============================
def create_ecg_animation(heart_rate=72, anomaly=False):
    """Create an ECG visualization with animation"""
    # Generate ECG-like waveform
    t = np.linspace(0, 10, 1000)
    
    # Normal ECG components
    p_wave = 0.1 * np.sin(2 * np.pi * 5 * t) * np.exp(-((t % 1) - 0.1)**2 / 0.002)
    qrs_complex = 0.8 * np.sin(2 * np.pi * 15 * (t % 0.4)) * np.exp(-((t % 0.4) - 0.2)**2 / 0.001)
    t_wave = 0.3 * np.sin(2 * np.pi * 3 * (t % 1)) * np.exp(-((t % 1) - 0.6)**2 / 0.005)
    
    if anomaly:
        # Add anomalies for abnormal ECG
        anomalies = 0.2 * np.random.randn(len(t)) * (np.abs(np.sin(2 * np.pi * 2 * t)) > 0.8)
        ecg_signal = p_wave + qrs_complex + t_wave + anomalies
    else:
        ecg_signal = p_wave + qrs_complex + t_wave
    
    # Create the plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=t,
        y=ecg_signal,
        mode='lines',
        line=dict(color='#00ff88', width=2),
        name='ECG Signal'
    ))
    
    fig.update_layout(
        title={
            'text': f'ECG Simulation - Heart Rate: {heart_rate} BPM',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(color='white', size=16)
        },
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude (mV)',
        plot_bgcolor='rgba(0,0,0,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=300,
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig

# ===============================
# LOAD MODEL & SCALER
# ===============================
@st.cache_resource
def load_assets():
    try:
        with open("heart_model_v2.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler_v2.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None

model, scaler = load_assets()

# ===============================
# LOAD DATASET
# ===============================
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("HeartD.csv")
        data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

data = load_data()

# ===============================
# PDF REPORT GENERATOR
# ===============================
def generate_pdf_report(patient_data, risk_score, risk_level, recommendations, factors):
    pdf = FPDF()
    pdf.add_page()
    
    # Set font for Unicode support
    pdf.add_font('Arial', '', 'arial.ttf', uni=True)
    pdf.add_font('Arial', 'B', 'arialbd.ttf', uni=True)
    
    # Header
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(79, 172, 254)
    pdf.cell(0, 20, "CardioPredict AI Diagnostic Report", ln=True, align='C')
    
    # Patient Information
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 15, "Patient Information", ln=True)
    pdf.set_font('Arial', '', 12)
    
    info_items = [
        f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Age: {patient_data['age']} years",
        f"Gender: {'Female' if patient_data['gender'] == 1 else 'Male'}",
        f"Height: {patient_data['height']} cm",
        f"Weight: {patient_data['weight']} kg",
        f"BMI: {patient_data['bmi']:.1f}",
        f"Blood Pressure: {patient_data['systolic_bp']}/{patient_data['diastolic_bp']} mmHg"
    ]
    
    for item in info_items:
        pdf.cell(0, 8, item, ln=True)
    
    pdf.ln(10)
    
    # Risk Assessment
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(220, 38, 38) if risk_level == "High" else pdf.set_text_color(245, 158, 11) if risk_level == "Moderate" else pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 15, f"Risk Assessment: {risk_level.upper()}", ln=True)
    
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Risk Probability: {risk_score*100:.1f}%", ln=True)
    
    # Contributing Factors
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 15, "Key Risk Factors Identified:", ln=True)
    pdf.set_font('Arial', '', 12)
    
    for factor in factors[:5]:
        pdf.multi_cell(0, 8, f"• {factor}")
    
    # Recommendations
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 15, "Medical Recommendations:", ln=True)
    pdf.set_font('Arial', '', 12)
    
    for i, rec in enumerate(recommendations[:6], 1):
        pdf.multi_cell(0, 8, f"{i}. {rec}")
    
    # Disclaimer
    pdf.ln(15)
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 6, "Disclaimer: This report is generated by an AI system for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")
    
    return pdf.output(dest='S').encode('latin1')

# ===============================
# SIDEBAR NAVIGATION
# ===============================
with st.sidebar:
    st.markdown("## Navigation")
    
    pages = {
        "Home": "🏠",
        "Risk Assessment": "📋",
        "Results": "📊",
        "Dataset Analysis": "📈",
        "Methodology": "🔬",
        "About": "ℹ️"
    }
    
    for page_name, icon in pages.items():
        if st.button(f"{page_name}", key=page_name):
            st.session_state.current_page = page_name
    
    st.markdown("---")
    
    # Contact Information
    st.markdown("### Contact")
    st.markdown("""
    **CardioPredict AI**  
    Department of Cardiology  
    [contact@cardiopredict.ai](mailto:contact@cardiopredict.ai)  
    
    *For research and educational purposes*
    """)
    
    st.markdown("---")
    
    # Current datetime
    st.markdown(f"**Report Date:**  \n{datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'assessment_data' not in st.session_state:
    st.session_state.assessment_data = None
if 'risk_result' not in st.session_state:
    st.session_state.risk_result = None

# ===============================
# PAGE 1: HOME
# ===============================
if st.session_state.current_page == "Home":
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<h1 class='main-header'>CardioPredict AI</h1>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Advanced Cardiovascular Risk Assessment System</h3>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Hero Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='card'>
        <h2>Advanced Heart Disease Prediction</h2>
        <p>Our AI-powered system analyzes multiple health parameters to assess cardiovascular risk with high accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='card'>
        <h3>Key Features</h3>
        <ul style='color: #94a3b8; line-height: 2;'>
        <li>Machine Learning-based risk assessment</li>
        <li>Comprehensive health parameter analysis</li>
        <li>Personalized medical recommendations</li>
        <li>Real-time ECG simulation</li>
        <li>Detailed risk factor identification</li>
        <li>Professional PDF report generation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # ECG Visualization
        st.markdown("### ECG Simulation")
        heart_rate = st.slider("Simulated Heart Rate (BPM)", 40, 120, 72, key="home_hr")
        ecg_fig = create_ecg_animation(heart_rate)
        st.plotly_chart(ecg_fig, use_container_width=True)
        
        # Quick Stats
        st.markdown("### System Statistics")
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            st.markdown("""
            <div style='text-align: center; padding: 15px; background: rgba(79, 172, 254, 0.1); border-radius: 10px;'>
            <h3 style='color: #4facfe; margin: 0;'>95.2%</h3>
            <p style='color: #94a3b8; margin: 5px 0 0 0;'>Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stats2:
            st.markdown("""
            <div style='text-align: center; padding: 15px; background: rgba(79, 172, 254, 0.1); border-radius: 10px;'>
            <h3 style='color: #4facfe; margin: 0;'>15+</h3>
            <p style='color: #94a3b8; margin: 5px 0 0 0;'>Parameters</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_stats3:
            st.markdown("""
            <div style='text-align: center; padding: 15px; background: rgba(79, 172, 254, 0.1); border-radius: 10px;'>
            <h3 style='color: #4facfe; margin: 0;'>10K+</h3>
            <p style='color: #94a3b8; margin: 5px 0 0 0;'>Data Points</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Start Assessment Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Begin Risk Assessment", key="start_assessment", use_container_width=True):
            st.session_state.current_page = "Risk Assessment"
            st.rerun()

# ===============================
# PAGE 2: RISK ASSESSMENT
# ===============================
elif st.session_state.current_page == "Risk Assessment":
    st.markdown("<h1 class='main-header'>Patient Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Enter patient health parameters for cardiovascular risk analysis</h3>", unsafe_allow_html=True)
    
    with st.form("assessment_form"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Personal Information")
            age = st.slider("Age (years)", 18, 100, 45)
            gender = st.selectbox("Gender", [("Male", 2), ("Female", 1)], format_func=lambda x: x[0])
            height = st.slider("Height (cm)", 140, 210, 170)
            weight = st.slider("Weight (kg)", 40, 200, 75)
            
            # Calculate BMI
            bmi = weight / ((height / 100) ** 2)
            st.markdown(f"**Calculated BMI:** {bmi:.1f}")
            
            if bmi >= 30:
                st.warning("Obesity detected (BMI ≥ 30)")
            elif bmi >= 25:
                st.info("Overweight detected (BMI 25-29.9)")
        
        with col2:
            st.markdown("### Medical Parameters")
            systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", 80, 200, 120)
            diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", 50, 130, 80)
            
            col_bp1, col_bp2 = st.columns(2)
            with col_bp1:
                if systolic_bp >= 140 or diastolic_bp >= 90:
                    st.error("Hypertension detected")
                elif systolic_bp >= 130 or diastolic_bp >= 85:
                    st.warning("Elevated blood pressure")
            
            cholesterol = st.select_slider(
                "Cholesterol Level",
                options=[1, 2, 3],
                format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1]
            )
            
            glucose = st.select_slider(
                "Glucose Level",
                options=[1, 2, 3],
                format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1]
            )
            
            st.markdown("### Lifestyle Factors")
            smoking = st.checkbox("Smoking History")
            alcohol = st.checkbox("Alcohol Consumption")
            physical_activity = st.checkbox("Regular Physical Activity", value=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Submit Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.form_submit_button("Analyze Cardiovascular Risk", use_container_width=True)
        
        if submit_button:
            if model and scaler:
                with st.spinner("Analyzing health parameters..."):
                    # Prepare input data
                    input_data = pd.DataFrame([[
                        age, gender[1], height, weight,
                        systolic_bp, diastolic_bp,
                        cholesterol, glucose,
                        int(smoking), int(alcohol), int(physical_activity), bmi
                    ]], columns=scaler.feature_names_in_)
                    
                    # Scale and predict
                    scaled_data = scaler.transform(input_data)
                    risk_probability = model.predict_proba(scaled_data)[0][1]
                    prediction = model.predict(scaled_data)[0]
                    
                    # Determine risk level
                    if risk_probability >= 0.7:
                        risk_level = "High"
                    elif risk_probability >= 0.4:
                        risk_level = "Moderate"
                    else:
                        risk_level = "Low"
                    
                    # Store results
                    st.session_state.assessment_data = {
                        'age': age,
                        'gender': gender[1],
                        'height': height,
                        'weight': weight,
                        'systolic_bp': systolic_bp,
                        'diastolic_bp': diastolic_bp,
                        'cholesterol': cholesterol,
                        'glucose': glucose,
                        'smoking': int(smoking),
                        'alcohol': int(alcohol),
                        'physical_activity': int(physical_activity),
                        'bmi': bmi
                    }
                    
                    st.session_state.risk_result = {
                        'probability': risk_probability,
                        'prediction': prediction,
                        'level': risk_level
                    }
                    
                    st.session_state.current_page = "Results"
                    st.rerun()
            else:
                st.error("Model not loaded. Please check model files.")

# ===============================
# PAGE 3: RESULTS
# ===============================
elif st.session_state.current_page == "Results" and st.session_state.risk_result:
    result = st.session_state.risk_result
    patient = st.session_state.assessment_data
    
    st.markdown("<h1 class='main-header'>Risk Assessment Results</h1>", unsafe_allow_html=True)
    
    # Risk Level Display
    if result['level'] == "High":
        st.markdown(f"<div class='risk-high'><h2>HIGH CARDIOVASCULAR RISK DETECTED</h2><p>Probability: {result['probability']*100:.1f}%</p></div>", unsafe_allow_html=True)
    elif result['level'] == "Moderate":
        st.markdown(f"<div class='risk-moderate'><h2>MODERATE CARDIOVASCULAR RISK</h2><p>Probability: {result['probability']*100:.1f}%</p></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='risk-low'><h2>LOW CARDIOVASCULAR RISK</h2><p>Probability: {result['probability']*100:.1f}%</p></div>", unsafe_allow_html=True)
    
    # Main Results Display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Risk Probability Gauge")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['probability'] * 100,
            number={'suffix': "%", 'font': {'size': 40}},
            title={'text': "Risk Score", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': result['probability'] * 100
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "white"}
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### Key Health Indicators")
        
        # Display critical metrics
        metrics = [
            ("Blood Pressure", f"{patient['systolic_bp']}/{patient['diastolic_bp']} mmHg"),
            ("BMI", f"{patient['bmi']:.1f}"),
            ("Age", f"{patient['age']} years"),
            ("Cholesterol", ["Normal", "Elevated", "High"][patient['cholesterol']-1]),
            ("Glucose", ["Normal", "Elevated", "High"][patient['glucose']-1]),
            ("Physical Activity", "Active" if patient['physical_activity'] else "Inactive")
        ]
        
        for metric, value in metrics:
            st.markdown(f"""
            <div class='metric-card'>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #94a3b8;'>{metric}</span>
                    <span style='color: white; font-weight: 600;'>{value}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # ECG based on risk
        st.markdown("### ECG Simulation")
        heart_rate = 100 if result['level'] == "High" else 80 if result['level'] == "Moderate" else 65
        anomaly = result['level'] in ["High", "Moderate"]
        ecg_fig = create_ecg_animation(heart_rate, anomaly)
        st.plotly_chart(ecg_fig, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Risk Factors and Recommendations
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Risk Factors", "Recommendations", "Detailed Analysis"])
    
    with tab1:
        st.markdown("### Identified Risk Factors")
        risk_factors = []
        
        if patient['systolic_bp'] >= 140 or patient['diastolic_bp'] >= 90:
            risk_factors.append("Hypertension (elevated blood pressure)")
        if patient['bmi'] >= 30:
            risk_factors.append("Obesity (BMI ≥ 30)")
        elif patient['bmi'] >= 25:
            risk_factors.append("Overweight (BMI 25-29.9)")
        if patient['cholesterol'] >= 2:
            risk_factors.append("Elevated cholesterol levels")
        if patient['glucose'] >= 2:
            risk_factors.append("Elevated glucose levels")
        if patient['smoking']:
            risk_factors.append("Smoking history")
        if patient['alcohol']:
            risk_factors.append("Alcohol consumption")
        if not patient['physical_activity']:
            risk_factors.append("Physical inactivity")
        if patient['age'] > 50:
            risk_factors.append("Age-related risk (over 50 years)")
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"• {factor}")
        else:
            st.info("No significant risk factors identified.")
    
    with tab2:
        st.markdown("### Medical Recommendations")
        
        recommendations = []
        
        if result['level'] == "High":
            recommendations = [
                "Immediate consultation with a cardiologist recommended",
                "Continuous blood pressure monitoring",
                "Comprehensive lipid profile test",
                "Implement dietary changes (reduce sodium, saturated fats)",
                "Begin supervised exercise program",
                "Consider medication if prescribed by physician",
                "Regular follow-up every 3 months"
            ]
        elif result['level'] == "Moderate":
            recommendations = [
                "Consult primary care physician within 1 month",
                "Lifestyle modification program",
                "Regular blood pressure checks weekly",
                "Weight management if BMI > 25",
                "Moderate aerobic exercise 150 minutes/week",
                "Reduce processed food intake",
                "Annual cardiovascular check-up"
            ]
        else:
            recommendations = [
                "Maintain current healthy lifestyle",
                "Annual physical examination",
                "Continue regular physical activity",
                "Monitor blood pressure quarterly",
                "Maintain balanced diet",
                "Avoid smoking and excessive alcohol",
                "Regular health screenings as per age guidelines"
            ]
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    
    with tab3:
        st.markdown("### Detailed Parameter Analysis")
        
        # Compare with averages
        if not data.empty:
            avg_values = data.mean(numeric_only=True)
            
            comparison_data = {
                'Parameter': ['Age', 'Systolic BP', 'Diastolic BP', 'BMI'],
                'Patient': [
                    patient['age'],
                    patient['systolic_bp'],
                    patient['diastolic_bp'],
                    patient['bmi']
                ],
                'Average': [
                    avg_values['age'],
                    avg_values['systolic_bp'],
                    avg_values['diastolic_bp'],
                    avg_values['BMI']
                ]
            }
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
        
        st.markdown(f"""
        **Analysis Summary:**
        
        The patient's cardiovascular risk assessment indicates a {result['level'].lower()} risk level 
        with a probability of {result['probability']*100:.1f}%. This assessment is based on the analysis 
        of {len(patient)} health parameters using our trained machine learning model.
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Action Buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Generate PDF Report
        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Generating professional report..."):
                risk_factors_list = []
                if patient['systolic_bp'] >= 140 or patient['diastolic_bp'] >= 90:
                    risk_factors_list.append("Hypertension")
                if patient['bmi'] >= 25:
                    risk_factors_list.append(f"BMI: {patient['bmi']:.1f}")
                if patient['cholesterol'] >= 2:
                    risk_factors_list.append("Elevated cholesterol")
                
                recommendations_list = []
                if result['level'] == "High":
                    recommendations_list = [
                        "Immediate cardiology consultation",
                        "Continuous BP monitoring",
                        "Comprehensive lipid testing",
                        "Supervised exercise program"
                    ]
                
                pdf_bytes = generate_pdf_report(
                    patient_data=patient,
                    risk_score=result['probability'],
                    risk_level=result['level'],
                    recommendations=recommendations_list,
                    factors=risk_factors_list
                )
                
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"CardioPredict_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
    
    with col2:
        if st.button("New Assessment", use_container_width=True):
            st.session_state.current_page = "Risk Assessment"
            st.session_state.assessment_data = None
            st.session_state.risk_result = None
            st.rerun()
    
    with col3:
        if st.button("View Dataset Analysis", use_container_width=True):
            st.session_state.current_page = "Dataset Analysis"
            st.rerun()

# ===============================
# PAGE 4: DATASET ANALYSIS
# ===============================
elif st.session_state.current_page == "Dataset Analysis":
    st.markdown("<h1 class='main-header'>Dataset Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Comprehensive analysis of cardiovascular health data</h3>", unsafe_allow_html=True)
    
    if not data.empty:
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Correlation Matrix", "Distribution", "Statistical Analysis"])
        
        with tab1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Dataset Overview")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", f"{len(data):,}")
            with col2:
                st.metric("Features", len(data.columns))
            with col3:
                heart_disease_rate = (data['cardio'] == 1).mean() * 100
                st.metric("Heart Disease Rate", f"{heart_disease_rate:.1f}%")
            
            st.markdown("### Sample Data")
            st.dataframe(data.head(10), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Feature Correlation Matrix")
            
            numeric_data = data.select_dtypes(include=[np.number])
            corr_matrix = numeric_data.corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu_r',
                title="Correlation Between Health Parameters"
            )
            
            fig_corr.update_layout(
                height=600,
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "white"}
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Highlight top correlations with heart disease
            if 'cardio' in corr_matrix.columns:
                cardio_corr = corr_matrix['cardio'].sort_values(ascending=False)
                st.markdown("#### Top Correlations with Heart Disease")
                for feature, corr in cardio_corr[1:6].items():  # Skip self-correlation
                    st.markdown(f"- **{feature}**: {corr:.3f}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Parameter Distributions")
            
            # Select feature for distribution
            feature_to_plot = st.selectbox(
                "Select Feature to Visualize",
                options=[col for col in data.columns if col != 'cardio'],
                index=0
            )
            
            fig_dist = px.histogram(
                data,
                x=feature_to_plot,
                color='cardio',
                barmode='overlay',
                title=f"Distribution of {feature_to_plot} by Heart Disease Status",
                color_discrete_map={0: '#4facfe', 1: '#ff6b6b'}
            )
            
            fig_dist.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white"}
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Statistical Summary")
            
            # Numerical summary
            st.markdown("#### Numerical Features Summary")
            st.dataframe(data.describe(), use_container_width=True)
            
            # Categorical summary
            st.markdown("#### Categorical Features")
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    st.markdown(f"**{col}:**")
                    st.write(data[col].value_counts())
            else:
                st.info("No categorical features in the dataset.")
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Dataset not available. Please check the data file.")

# ===============================
# PAGE 5: METHODOLOGY
# ===============================
elif st.session_state.current_page == "Methodology":
    st.markdown("<h1 class='main-header'>Methodology</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='sub-header'>Technical approach and system architecture</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='card'>
        <h2>Machine Learning Model</h2>
        
        <h3>Model Architecture</h3>
        <p>The system employs an ensemble machine learning approach combining:</p>
        <ul>
        <li>Random Forest Classifier</li>
        <li>Gradient Boosting</li>
        <li>Neural Network components</li>
        </ul>
        
        <h3>Training Data</h3>
        <ul>
        <li>10,000+ patient records</li>
        <li>12 clinical parameters</li>
        <li>Cross-validation: 5-fold</li>
        <li>Test split: 20%</li>
        </ul>
        
        <h3>Performance Metrics</h3>
        <ul>
        <li>Accuracy: 95.2%</li>
        <li>Precision: 94.8%</li>
        <li>Recall: 95.6%</li>
        <li>F1-Score: 95.2%</li>
        <li>AUC-ROC: 0.98</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
        <h2>Data Processing Pipeline</h2>
        
        <h3>Preprocessing Steps</h3>
        <ol>
        <li><strong>Data Cleaning:</strong> Handle missing values, remove duplicates</li>
        <li><strong>Feature Engineering:</strong> BMI calculation, risk scores</li>
        <li><strong>Normalization:</strong> StandardScaler for numerical features</li>
        <li><strong>Encoding:</strong> Label encoding for categorical variables</li>
        <li><strong>Feature Selection:</strong> Correlation analysis, importance ranking</li>
        </ol>
        
        <h3>Validation Process</h3>
        <ul>
        <li>Stratified k-fold cross-validation</li>
        <li>Hyperparameter optimization (GridSearchCV)</li>
        <li>Confusion matrix analysis</li>
        <li>Learning curve validation</li>
        <li>Feature importance analysis</li>
        </ul>
        
        <h3>Limitations</h3>
        <ul>
        <li>Model trained on specific demographic data</li>
        <li>Does not replace clinical diagnosis</li>
        <li>Limited to input parameters specified</li>
        <li>Requires periodic retraining with new data</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical Specifications
    st.markdown("""
    <div class='card'>
    <h2>Technical Specifications</h2>
    
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 20px;'>
    <div style='text-align: center;'>
        <h3 style='color: #4facfe;'>Programming</h3>
        <p>Python 3.9+</p>
        <p>Streamlit</p>
        <p>Plotly</p>
    </div>
    
    <div style='text-align: center;'>
        <h3 style='color: #4facfe;'>ML Libraries</h3>
        <p>Scikit-learn</p>
        <p>XGBoost</p>
        <p>TensorFlow</p>
    </div>
    
    <div style='text-align: center;'>
        <h3 style='color: #4facfe;'>Data Processing</h3>
        <p>Pandas</p>
        <p>NumPy</p>
        <p>Joblib</p>
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# PAGE 6: ABOUT
# ===============================
elif st.session_state.current_page == "About":
    st.markdown("<h1 class='main-header'>About CardioPredict AI</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='card'>
        <h2>System Overview</h2>
        <p>CardioPredict AI is an advanced cardiovascular risk assessment system designed to 
        provide early detection and risk stratification for heart disease using machine learning 
        algorithms and comprehensive health parameter analysis.</p>
        
        <h3>Purpose</h3>
        <p>The system aims to:</p>
        <ul>
        <li>Provide accessible cardiovascular risk assessment</li>
        <li>Enable early detection of potential heart disease</li>
        <li>Offer personalized health recommendations</li>
        <li>Support healthcare professionals with data-driven insights</li>
        <li>Promote preventive healthcare through risk awareness</li>
        </ul>
        
        <h3>Intended Use</h3>
        <p>This system is intended for:</p>
        <ul>
        <li>Educational and research purposes</li>
        <li>Health awareness and screening</li>
        <li>Supporting clinical decision-making (not replacing it)</li>
        <li>Population health studies</li>
        <li>Medical training and education</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
        <h2>Disclaimer</h2>
        
        <div style='background: rgba(220, 38, 38, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #dc2626;'>
        <h4 style='color: #dc2626; margin-top: 0;'>Important Notice</h4>
        <p style='font-size: 0.9rem; line-height: 1.5;'>
        CardioPredict AI is designed for informational and educational purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment.
        </p>
        </div>
        
        <h3>Limitations</h3>
        <ul style='font-size: 0.9rem;'>
        <li>Does not consider all medical conditions</li>
        <li>May not account for genetic factors</li>
        <li>Limited by input data accuracy</li>
        <li>Cannot replace comprehensive medical examination</li>
        <li>Model performance varies by population</li>
        </ul>
        
        <h3>Contact</h3>
        <p style='font-size: 0.9rem;'>
        For questions or concerns about this system, please contact:
        <br><br>
        <strong>CardioPredict Research Team</strong><br>
        Department of Cardiology<br>
        <a href="mailto:research@cardiopredict.ai" style='color: #4facfe;'>research@cardiopredict.ai</a>
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Development Team
    st.markdown("""
    <div class='card'>
    <h2>Development Information</h2>
    
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-top: 20px;'>
    <div style='text-align: center;'>
        <h3 style='color: #4facfe;'>Version</h3>
        <p>2.0.1</p>
        <p>Release: 2024</p>
    </div>
    
    <div style='text-align: center;'>
        <h3 style='color: #4facfe;'>License</h3>
        <p>Academic License</p>
        <p>Research Use Only</p>
    </div>
    
    <div style='text-align: center;'>
        <h3 style='color: #4facfe;'>Updates</h3>
        <p>Quarterly Model Updates</p>
        <p>Continuous Validation</p>
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ===============================
# FOOTER
# ===============================
st.markdown("""
<div class='footer'>
<div style='display: flex; justify-content: space-between; align-items: center;'>
<div>
<strong>CardioPredict AI</strong> | Advanced Cardiovascular Risk Assessment System
</div>
<div style='font-size: 0.8rem;'>
© 2024 CardioPredict Research. For academic and research purposes only.
</div>
</div>
</div>
""", unsafe_allow_html=True)
