
# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # -------------------------------
# # Page configuration
# # -------------------------------
# st.set_page_config(
#     page_title="Heart Disease Prediction",
#     page_icon="❤️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Apply custom CSS
# st.markdown("""
# <style>
#     @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
#     * {
#         font-family: 'Poppins', sans-serif;
#     }
    
#     .main-header {
#         text-align: center;
#         padding: 1rem;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         border-radius: 10px;
#         color: white;
#         margin-bottom: 2rem;
#     }
    
#     .metric-card {
#         background: white;
#         padding: 1.5rem;
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         margin: 1rem 0;
#         border-left: 5px solid #667eea;
#     }
    
#     .risk-high {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
#         color: white;
#         padding: 2rem;
#         border-radius: 15px;
#         text-align: center;
#         animation: pulse 2s infinite;
#     }
    
#     .risk-low {
#         background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
#         color: white;
#         padding: 2rem;
#         border-radius: 15px;
#         text-align: center;
#     }
    
#     @keyframes pulse {
#         0% { transform: scale(1); }
#         50% { transform: scale(1.02); }
#         100% { transform: scale(1); }
#     }
    
#     .stButton>button {
#         width: 100%;
#         background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#         color: white;
#         border: none;
#         padding: 0.75rem 1.5rem;
#         font-weight: 600;
#         border-radius: 8px;
#         transition: all 0.3s ease;
#     }
    
#     .stButton>button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
#     }
    
#     .sidebar .sidebar-content {
#         background: linear-gradient(180deg, #f5f7fa 0%, #c3cfe2 100%);
#     }
    
#     .stNumberInput>div>div>input, .stSelectbox>div>div>select {
#         border-radius: 8px;
#     }
    
#     .feature-input {
#         margin-bottom: 1rem;
#     }
    
#     .prediction-result {
#         font-size: 1.5rem;
#         font-weight: bold;
#         margin: 1rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # -------------------------------
# # Load model and scaler
# # -------------------------------
# with open("heart_model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# # -------------------------------
# # Header
# # -------------------------------
# col1, col2, col3 = st.columns([1, 2, 1])
# with col2:
#     st.markdown('<div class="main-header"><h1>❤️ Heart Disease Prediction System</h1><p>AI-Powered Cardiac Health Assessment</p></div>', unsafe_allow_html=True)

# # -------------------------------
# # Sidebar Inputs with improved layout
# # -------------------------------
# with st.sidebar:
#     st.markdown("### 🏥 Patient Information")
#     st.markdown("---")
    
#     # Personal Information
#     st.markdown("##### 👤 Personal Details")
#     col1, col2 = st.columns(2)
#     with col1:
#         age = st.number_input("Age (years)", 18, 100, 45, help="Patient's age in years")
#     with col2:
#         gender = st.selectbox(
#             "Gender",
#             [1, 2],
#             format_func=lambda x: "👩 Female" if x == 1 else "👨 Male",
#             help="Patient's gender"
#         )
    
#     # Physical Measurements
#     st.markdown("##### 📏 Physical Measurements")
#     col1, col2 = st.columns(2)
#     with col1:
#         height = st.number_input("Height (cm)", 140, 200, 165, help="Height in centimeters")
#     with col2:
#         weight = st.number_input("Weight (kg)", 40, 150, 70, help="Weight in kilograms")
    
#     st.markdown('<div class="feature-input">', unsafe_allow_html=True)
#     BMI = st.slider("BMI", 10.0, 60.0, 25.0, 0.1, help="Body Mass Index")
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Blood Pressure
#     st.markdown("##### 💓 Blood Pressure")
#     col1, col2 = st.columns(2)
#     with col1:
#         systolic_bp = st.number_input("Systolic BP", 80, 200, 120, help="Systolic blood pressure")
#     with col2:
#         diastolic_bp = st.number_input("Diastolic BP", 50, 130, 80, help="Diastolic blood pressure")
    
#     # Health Indicators
#     st.markdown("##### 🩺 Health Indicators")
#     cholesterol = st.selectbox(
#         "Cholesterol Level", 
#         [1, 2, 3],
#         format_func=lambda x: {1: "Normal 🟢", 2: "Above Normal 🟡", 3: "Well Above Normal 🔴"}[x],
#         help="Cholesterol level"
#     )
    
#     gluc = st.selectbox(
#         "Glucose Level", 
#         [1, 2, 3],
#         format_func=lambda x: {1: "Normal 🟢", 2: "Above Normal 🟡", 3: "Well Above Normal 🔴"}[x],
#         help="Glucose level"
#     )
    
#     # Lifestyle Factors
#     st.markdown("##### 🏃 Lifestyle Factors")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         smoke = st.selectbox(
#             "Smoking", 
#             [0, 1],
#             format_func=lambda x: "✅ Smoker" if x == 1 else "❌ Non-smoker"
#         )
#     with col2:
#         alco = st.selectbox(
#             "Alcohol", 
#             [0, 1],
#             format_func=lambda x: "✅ Drinks" if x == 1 else "❌ Doesn't drink"
#         )
#     with col3:
#         active = st.selectbox(
#             "Activity", 
#             [0, 1],
#             format_func=lambda x: "✅ Active" if x == 1 else "❌ Not Active"
#         )
    
#     st.markdown("---")
#     predict_button = st.button("🔍 Predict Heart Disease Risk", use_container_width=True)

# # -------------------------------
# # Main Content Area
# # -------------------------------
# if predict_button:
#     # Build input dict
#     input_dict = {
#         'age': age ,
#         'gender': gender,
#         'height': height,
#         'weight': weight,
#         'systolic_bp': systolic_bp,
#         'diastolic_bp': diastolic_bp,
#         'cholesterol': cholesterol,
#         'gluc': gluc,
#         'smoke': smoke,
#         'alco': alco,
#         'active': active,
#         'BMI': BMI
#     }

#     # Create DataFrame
#     input_df = pd.DataFrame(
#         [[input_dict[col] for col in scaler.feature_names_in_]],
#         columns=scaler.feature_names_in_
#     )

#     # Scale & predict
#     input_scaled = scaler.transform(input_df)
#     prediction = model.predict(input_scaled)[0]
#     prediction_proba = model.predict_proba(input_scaled)[0]

#     # Display Results
#     st.markdown("## 📊 Prediction Results")
#     st.markdown("---")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("### 🎯 Risk Assessment")
#         if prediction == 1:
#             st.markdown('<div class="risk-high">', unsafe_allow_html=True)
#             st.markdown("### ⚠️ HIGH RISK")
#             st.markdown("### Heart Disease Detected")
#             st.markdown(f"Probability: **{prediction_proba[1]*100:.1f}%**")
#             st.markdown('</div>', unsafe_allow_html=True)
#             st.warning("**Recommendation:** Please consult a cardiologist immediately.")
#         else:
#             st.markdown('<div class="risk-low">', unsafe_allow_html=True)
#             st.markdown("### ✅ LOW RISK")
#             st.markdown("### No Heart Disease Detected")
#             st.markdown(f"Probability: **{prediction_proba[0]*100:.1f}%**")
#             st.markdown('</div>', unsafe_allow_html=True)
#             st.success("**Recommendation:** Maintain a healthy lifestyle with regular check-ups.")
    
#     with col2:
#         st.markdown("### 📈 Risk Probability")
#         # Create a progress bar for visualization
#         risk_percentage = prediction_proba[1] * 100
        
#         fig, ax = plt.subplots(figsize=(8, 1))
#         ax.barh([0], [100], color='lightgray', alpha=0.3)
#         ax.barh([0], [risk_percentage], 
#                 color='#f5576c' if prediction == 1 else '#4facfe',
#                 height=0.5)
#         ax.set_xlim(0, 100)
#         ax.set_xticks([0, 25, 50, 75, 100])
#         ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
#         ax.set_yticks([])
#         ax.set_title(f"Risk Score: {risk_percentage:.1f}%", fontsize=14, fontweight='bold')
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['left'].set_visible(False)
#         st.pyplot(fig)
        
#         # Probability breakdown
#         st.markdown('<div class="metric-card">', unsafe_allow_html=True)
#         st.markdown("##### Probability Breakdown")
#         col_prob1, col_prob2 = st.columns(2)
#         with col_prob1:
#             st.metric("Low Risk", f"{prediction_proba[0]*100:.1f}%")
#         with col_prob2:
#             st.metric("High Risk", f"{prediction_proba[1]*100:.1f}%")
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     st.markdown("---")
    
#     # -------------------------------
#     # Model Performance Section
#     # -------------------------------
#     st.markdown("## 📊 Model Performance Metrics")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown('<div class="metric-card">', unsafe_allow_html=True)
#         st.metric("Model Accuracy", "72%", "±2%")
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     with col2:
#         st.markdown('<div class="metric-card">', unsafe_allow_html=True)
#         st.metric("Precision", "75%")
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     with col3:
#         st.markdown('<div class="metric-card">', unsafe_allow_html=True)
#         st.metric("Recall", "70%")
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     # -------------------------------
#     # Confusion Matrix
#     # -------------------------------
#     st.markdown("### 🔍 Confusion Matrix")
    
#     # Load data and calculate confusion matrix
#     data = pd.read_csv("HeartD.csv")
#     data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

#     X = data.drop(['cardio', 'id'], axis=1)
#     y = data['cardio']

#     X_scaled = scaler.transform(X)
#     y_pred = model.predict(X_scaled)
#     cm = confusion_matrix(y, y_pred)

#     # Create styled confusion matrix
#     fig, ax = plt.subplots(figsize=(8, 6))
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low Risk', 'High Risk'])
#     disp.plot(cmap='Blues', ax=ax, values_format='d')
#     ax.set_title('Model Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
#     ax.set_xlabel('Predicted Label', fontsize=12)
#     ax.set_ylabel('True Label', fontsize=12)
#     st.pyplot(fig)
    
#     # Add explanation
#     with st.expander("📖 Understanding the Confusion Matrix"):
#         st.markdown("""
#         **What does this mean?**
#         - **True Negative (Top-left):** Correctly predicted as Low Risk
#         - **False Positive (Top-right):** Incorrectly predicted as High Risk (Type I error)
#         - **False Negative (Bottom-left):** Incorrectly predicted as Low Risk (Type II error)
#         - **True Positive (Bottom-right):** Correctly predicted as High Risk
        
#         **Note:** A good model has high values on the diagonal (top-left and bottom-right).
#         """)
    
#     st.markdown("---")
#     st.markdown("> **Disclaimer:** This tool is for educational purposes only. Always consult with a healthcare professional for medical advice.")

# else:
#     # Display welcome message when no prediction has been made
#     col1, col2, col3 = st.columns([1, 2, 1])
#     with col2:
#         st.markdown("""
#         <div style='text-align: center; padding: 3rem;'>
#             <h2>👋 Welcome to the Heart Disease Prediction System</h2>
#             <p style='font-size: 1.2rem; color: #666;'>
#                 This AI-powered tool helps assess your risk of cardiovascular disease 
#                 based on medical and lifestyle factors.
#             </p>
#             <br>
#             <h4>📝 How to use:</h4>
#             <p>1. Fill in your details in the sidebar</p>
#             <p>2. Click the <strong>'Predict Heart Disease Risk'</strong> button</p>
#             <p>3. View your personalized risk assessment</p>
#             <br>
#             <div style='background: #f0f2f6; padding: 2rem; border-radius: 10px;'>
#                 <h5>🔒 Your privacy matters:</h5>
#                 <p>All data is processed locally and not stored anywhere.</p>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Feature importance/description
#     st.markdown("---")
#     st.markdown("### 📋 Features Used in Prediction")
    
#     features_info = {
#         "Demographic": ["Age", "Gender", "BMI"],
#         "Vital Signs": ["Systolic BP", "Diastolic BP"],
#         "Blood Tests": ["Cholesterol", "Glucose"],
#         "Lifestyle": ["Smoking", "Alcohol", "Physical Activity"]
#     }
    
#     cols = st.columns(4)
#     for idx, (category, features) in enumerate(features_info.items()):
#         with cols[idx]:
#             st.markdown(f'<div class="metric-card"><h4>{category}</h4>', unsafe_allow_html=True)
#             for feature in features:
#                 st.markdown(f"• {feature}")
#             st.markdown('</div>', unsafe_allow_html=True)


import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -------------------------------
# Page configuration (UNCHANGED)
# -------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# -------------------------------
# Load model and scaler
# -------------------------------
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# Title (UNCHANGED)
# -------------------------------
st.title("❤️ Heart Disease Prediction System")
st.write("ML_Project_HeartDisease – Deployed ML Application")
st.markdown("---")

# -------------------------------
# Sidebar Inputs (UNCHANGED UI)
# -------------------------------
st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age (years)", 18, 100, 45)

gender = st.sidebar.selectbox(
    "Gender",
    [1, 2],
    format_func=lambda x: "Female" if x == 1 else "Male"
)

height = st.sidebar.number_input("Height (cm)", 140, 200, 165)
weight = st.sidebar.number_input("Weight (kg)", 40, 150, 70)

systolic_bp = st.sidebar.number_input("Systolic BP", 80, 200, 120)
diastolic_bp = st.sidebar.number_input("Diastolic BP", 50, 130, 80)

cholesterol = st.sidebar.selectbox("Cholesterol Level", [1, 2, 3])
gluc = st.sidebar.selectbox("Glucose Level", [1, 2, 3])

smoke = st.sidebar.selectbox("Smoking", [0, 1])
alco = st.sidebar.selectbox("Alcohol Intake", [0, 1])
active = st.sidebar.selectbox("Physical Activity", [0, 1])

predict = st.sidebar.button("Predict")

# -------------------------------
# Prediction
# -------------------------------
if predict:

    # ✅ BMI AUTO-CALCULATED (NO UI CHANGE)
    BMI = weight / ((height / 100) ** 2)

    # Build input dictionary (EXACT training features)
    input_dict = {
        'age': age,
        'gender': gender,
        'height': height,
        'weight': weight,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': smoke,
        'alco': alco,
        'active': active,
        'BMI': BMI
    }

    # Create DataFrame in EXACT training order
    input_df = pd.DataFrame(
        [[input_dict[col] for col in scaler.feature_names_in_]],
        columns=scaler.feature_names_in_
    )

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.markdown("---")

    # -------------------------------
    # Model Performance (UNCHANGED)
    # -------------------------------
    st.subheader("Model Performance")
    st.info("Model Accuracy: **72%**")

    # -------------------------------
    # Confusion Matrix (UNCHANGED)
    # -------------------------------
    data = pd.read_csv("HeartD.csv")
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

    X = data.drop(['cardio', 'id'], axis=1)
    y = data['cardio']

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low Risk", "High Risk"]
    ).plot(ax=ax, cmap="Blues", values_format="d")

    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
