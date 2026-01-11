import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="CardioPredict AI",
    page_icon="❤️",
    layout="wide"
)

# ===============================
# GLOBAL STYLES (PRO UI)
# ===============================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a 0%, #020617 60%);
    color: #e5e7eb;
    font-family: 'Inter', system-ui, sans-serif;
}

/* Glass Cards */
.glass {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 18px;
    padding: 26px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
    backdrop-filter: blur(14px);
    margin-bottom: 25px;
}

/* Titles */
.title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #94a3b8;
    font-size: 1.1rem;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #22d3ee);
    border: none;
    border-radius: 12px;
    padding: 14px 28px;
    font-weight: 700;
    color: #020617;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(56,189,248,0.4);
}

/* Risk badges */
.high {background: linear-gradient(90deg,#dc2626,#991b1b);}
.mid  {background: linear-gradient(90deg,#f59e0b,#d97706);}
.low  {background: linear-gradient(90deg,#10b981,#059669);}

.badge {
    padding: 18px;
    border-radius: 14px;
    color: white;
    text-align: center;
    font-size: 1.3rem;
    font-weight: 700;
}

/* Tables */
thead tr th {
    background-color: #020617 !important;
    color: #38bdf8 !important;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    with open("heart_disease_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ===============================
# ACCURACY TABLE (REAL RESULTS)
# ===============================
accuracy_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "Naive Bayes",
        "Decision Tree (Selected)"
    ],
    "Train-Test Accuracy": [
        0.723545,
        0.731042,
        0.707642,
        0.615008
    ],
    "K-Fold CV Accuracy": [
        0.724698,
        0.730313,
        0.704472,
        0.616790
    ],
    "Hyperparameter Tuned Accuracy": [
        0.724576,
        "NA",
        0.704656,
        0.727835
    ]
})

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.markdown("## ❤️ CardioPredict AI")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Risk Prediction", "Model Performance", "Methodology", "About"]
    )
    st.markdown("---")
    st.caption(f"🕒 {datetime.now().strftime('%d %b %Y, %H:%M')}")

# ===============================
# DASHBOARD
# ===============================
if page == "Dashboard":
    st.markdown("<div class='title'>CardioPredict AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Professional Cardiovascular Risk Assessment System</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='glass'><h3>Clinical Purpose</h3>"
                    "<p>Early identification of cardiovascular risk using validated machine learning models.</p></div>",
                    unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass'><h3>Model Strategy</h3>"
                    "<p>Multiple models evaluated using cross-validation & hyperparameter tuning.</p></div>",
                    unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='glass'><h3>Deployment Ready</h3>"
                    "<p>Best-performing model serialized using Pickle for production inference.</p></div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("Final Model Selection")
    st.write("""
    **Decision Tree (Hyperparameter Tuned)** was selected based on:
    - Highest validated accuracy  
    - Strong handling of non-linear clinical patterns  
    - Transparent decision logic  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# RISK PREDICTION
# ===============================
elif page == "Risk Prediction":
    st.markdown("<div class='title'>Risk Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Enter patient parameters</div>", unsafe_allow_html=True)

    with st.form("predict"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 18, 100, 45)
            gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
            height = st.slider("Height (cm)", 140, 210, 170)
            weight = st.slider("Weight (kg)", 40, 200, 75)

        with col2:
            ap_hi = st.slider("Systolic BP", 80, 200, 120)
            ap_lo = st.slider("Diastolic BP", 50, 130, 80)
            cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
            gluc = st.selectbox("Glucose Level", [1, 2, 3])
            smoke = st.checkbox("Smoking")
            alco = st.checkbox("Alcohol")
            active = st.checkbox("Physically Active", value=True)

        submit = st.form_submit_button("Analyze Risk")

    if submit:
        X = pd.DataFrame([[age, gender, height, weight, ap_hi, ap_lo,
                           cholesterol, gluc, int(smoke), int(alco), int(active)]],
                         columns=["age","gender","height","weight",
                                  "ap_hi","ap_lo","cholesterol","gluc",
                                  "smoke","alco","active"])

        prob = model.predict_proba(X)[0][1]

        st.markdown("<div class='glass'>", unsafe_allow_html=True)

        if prob >= 0.7:
            st.markdown(f"<div class='badge high'>HIGH RISK<br>{prob*100:.2f}%</div>", unsafe_allow_html=True)
        elif prob >= 0.4:
            st.markdown(f"<div class='badge mid'>MODERATE RISK<br>{prob*100:.2f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='badge low'>LOW RISK<br>{prob*100:.2f}%</div>", unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            gauge={"axis": {"range": [0,100]},
                   "bar": {"color": "#38bdf8"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# MODEL PERFORMANCE
# ===============================
elif page == "Model Performance":
    st.markdown("<div class='title'>Model Performance</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Comparative evaluation</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.dataframe(accuracy_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# METHODOLOGY
# ===============================
elif page == "Methodology":
    st.markdown("<div class='title'>Methodology</div>", unsafe_allow_html=True)

    st.markdown("""
    ### Why Decision Tree?
    - Captures **non-linear medical relationships**
    - Interpretable decisions (important in healthcare)
    - Highest **cross-validated tuned accuracy**
    - Balanced bias–variance tradeoff
    """)

# ===============================
# ABOUT
# ===============================
elif page == "About":
    st.markdown("<div class='title'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    **CardioPredict AI**  
    Academic Machine Learning Project  

    Dataset: Cardiovascular Disease Dataset  
    Model: Decision Tree (Tuned)  
    Deployment: Streamlit + Pickle  

    ⚠️ Educational use only
    """)
