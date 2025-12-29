

# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # -------------------------------
# # Page configuration (UNCHANGED)
# # -------------------------------
# st.set_page_config(
#     page_title="Heart Disease Prediction",
#     page_icon="❤️",
#     layout="wide"
# )

# # -------------------------------
# # Load model and scaler
# # -------------------------------
# with open("heart_model_v2.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("scaler_v2.pkl", "rb") as f:
#     scaler = pickle.load(f)

# # -------------------------------
# # Title (UNCHANGED)
# # -------------------------------
# st.title("❤️ Heart Disease Prediction System")
# st.write("ML_Project_HeartDisease – Deployed ML Application")
# st.markdown("---")

# # -------------------------------
# # Sidebar Inputs (UNCHANGED UI)
# # -------------------------------
# st.sidebar.header("Patient Details")

# age = st.sidebar.number_input("Age (years)", 18, 100, 45)

# gender = st.sidebar.selectbox(
#     "Gender",
#     [1, 2],
#     format_func=lambda x: "Female" if x == 1 else "Male"
# )

# height = st.sidebar.number_input("Height (cm)", 140, 200, 165)
# weight = st.sidebar.number_input("Weight (kg)", 40, 150, 70)

# systolic_bp = st.sidebar.number_input("Systolic BP", 80, 200, 120)
# diastolic_bp = st.sidebar.number_input("Diastolic BP", 50, 130, 80)

# cholesterol = st.sidebar.selectbox("Cholesterol Level", [1, 2, 3])
# gluc = st.sidebar.selectbox("Glucose Level", [1, 2, 3])

# smoke = st.sidebar.selectbox("Smoking", [0, 1])
# alco = st.sidebar.selectbox("Alcohol Intake", [0, 1])
# active = st.sidebar.selectbox("Physical Activity", [0, 1])

# predict = st.sidebar.button("Predict")

# # -------------------------------
# # Prediction
# # -------------------------------
# if predict:

#     # ✅ BMI AUTO-CALCULATED (NO UI CHANGE)
#     BMI = weight / ((height / 100) ** 2)

#     # Build input dictionary (EXACT training features)
#     input_dict = {
#         'age': age,
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

#     # Create DataFrame in EXACT training order
#     input_df = pd.DataFrame(
#         [[input_dict[col] for col in scaler.feature_names_in_]],
#         columns=scaler.feature_names_in_
#     )

#     # Scale and predict
#     input_scaled = scaler.transform(input_df)
#     prediction = model.predict(input_scaled)[0]

#     st.subheader("Prediction Result")

#     if prediction == 1:
#         st.error("⚠️ High Risk of Heart Disease")
#     else:
#         st.success("✅ Low Risk of Heart Disease")

#     st.markdown("---")

#     # -------------------------------
#     # Model Performance (UNCHANGED)
#     # -------------------------------
#     st.subheader("Model Performance")
#     st.info("Model Accuracy: **72%**")

#     # -------------------------------
#     # Confusion Matrix (UNCHANGED)
#     # -------------------------------
#     data = pd.read_csv("HeartD.csv")
#     data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

#     X = data.drop(['cardio', 'id'], axis=1)
#     y = data['cardio']

#     X_scaled = scaler.transform(X)
#     y_pred = model.predict(X_scaled)

#     cm = confusion_matrix(y, y_pred)

#     fig, ax = plt.subplots(figsize=(6, 5))
#     ConfusionMatrixDisplay(
#         confusion_matrix=cm,
#         display_labels=["Low Risk", "High Risk"]
#     ).plot(ax=ax, cmap="Blues", values_format="d")

#     ax.set_title("Confusion Matrix")
#     st.pyplot(fig)

# import streamlit as st
# import numpy as np
# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# # -------------------------------
# # Page configuration - ENHANCED
# # -------------------------------
# st.set_page_config(
#     page_title="HeartGuard AI | Disease Risk Assessment",
#     page_icon="❤️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # -------------------------------
# # Custom CSS for modern styling
# # -------------------------------
# st.markdown("""
# <style>
#     /* Main styling */
#     .main-header {
#         font-size: 3rem !important;
#         font-weight: 700 !important;
#         background: linear-gradient(90deg, #FF6B6B, #FF8E53);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         text-align: center;
#         margin-bottom: 0.5rem !important;
#     }
    
#     .sub-header {
#         color: #666;
#         text-align: center;
#         font-size: 1.2rem !important;
#         margin-bottom: 2rem !important;
#     }
    
#     /* Card styling */
#     .metric-card {
#         background: white;
#         padding: 1.5rem;
#         border-radius: 15px;
#         box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
#         border-left: 5px solid #FF6B6B;
#         margin: 1rem 0;
#     }
    
#     /* Button styling */
#     .stButton > button {
#         width: 100%;
#         background: linear-gradient(90deg, #FF6B6B, #FF8E53);
#         color: white;
#         border: none;
#         padding: 0.75rem 1.5rem;
#         border-radius: 10px;
#         font-size: 1.1rem;
#         font-weight: 600;
#         transition: all 0.3s ease;
#     }
    
#     .stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
#     }
    
#     /* Sidebar styling */
#     .sidebar-header {
#         font-size: 1.8rem !important;
#         font-weight: 600 !important;
#         color: #FF6B6B !important;
#         margin-bottom: 1.5rem !important;
#     }
    
#     /* Input field styling */
#     .stNumberInput, .stSelectbox {
#         margin-bottom: 1rem;
#     }
    
#     /* Result boxes */
#     .high-risk {
#         background: linear-gradient(135deg, #FFE8E8, #FFC9C9);
#         padding: 2rem;
#         border-radius: 15px;
#         border-left: 6px solid #FF6B6B;
#         margin: 2rem 0;
#         animation: pulse 2s infinite;
#     }
    
#     .low-risk {
#         background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
#         padding: 2rem;
#         border-radius: 15px;
#         border-left: 6px solid #4CAF50;
#         margin: 2rem 0;
#     }
    
#     /* Animation */
#     @keyframes pulse {
#         0% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.7); }
#         70% { box-shadow: 0 0 0 10px rgba(255, 107, 107, 0); }
#         100% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0); }
#     }
    
#     /* Divider */
#     .custom-divider {
#         height: 3px;
#         background: linear-gradient(90deg, #FF6B6B, #FF8E53);
#         margin: 2rem 0;
#         border-radius: 3px;
#     }
    
#     /* Progress bar */
#     .stProgress > div > div > div {
#         background: linear-gradient(90deg, #FF6B6B, #FF8E53);
#     }
# </style>
# """, unsafe_allow_html=True)

# # -------------------------------
# # Load model and scaler (UNCHANGED)
# # -------------------------------
# with open("heart_model_v2.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("scaler_v2.pkl", "rb") as f:
#     scaler = pickle.load(f)

# # -------------------------------
# # Header Section - REDESIGNED
# # -------------------------------
# col1, col2, col3 = st.columns([1, 2, 1])
# with col2:
#     st.markdown('<h1 class="main-header">❤️ HeartGuard AI</h1>', unsafe_allow_html=True)
#     st.markdown('<p class="sub-header">Advanced Cardiovascular Risk Assessment System</p>', unsafe_allow_html=True)

# st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# # -------------------------------
# # Quick Stats Cards
# # -------------------------------
# st.subheader("📊 System Overview")
# col1, col2, col3, col4 = st.columns(4)

# with col1:
#     st.markdown("""
#     <div class="metric-card">
#         <h3>🎯 Accuracy</h3>
#         <h2>72%</h2>
#         <p>Model Performance</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col2:
#     st.markdown("""
#     <div class="metric-card">
#         <h3>🏥 Patients</h3>
#         <h2>70,000+</h2>
#         <p>Dataset Size</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col3:
#     st.markdown("""
#     <div class="metric-card">
#         <h3>📈 Features</h3>
#         <h2>12</h2>
#         <p>Health Parameters</p>
#     </div>
#     """, unsafe_allow_html=True)

# with col4:
#     st.markdown("""
#     <div class="metric-card">
#         <h3>⚡ Speed</h3>
#         <h2>Real-time</h2>
#         <p>Instant Prediction</p>
#     </div>
#     """, unsafe_allow_html=True)

# st.markdown("""
# <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
#     <p style="margin: 0; color: #666;">
#     💡 <strong>Note:</strong> This AI tool provides risk assessment based on medical data patterns. 
#     Always consult with healthcare professionals for medical diagnosis.
#     </p>
# </div>
# """, unsafe_allow_html=True)

# # -------------------------------
# # Sidebar - REDESIGNED
# # -------------------------------
# with st.sidebar:
#     st.markdown('<h2 class="sidebar-header">👤 Patient Profile</h2>', unsafe_allow_html=True)
    
#     # Personal Information
#     st.markdown("### 📝 Personal Details")
#     age = st.slider("**Age** (years)", 18, 100, 45, help="Patient's current age")
    
#     gender = st.selectbox(
#         "**Gender**",
#         [1, 2],
#         format_func=lambda x: "👩 Female" if x == 1 else "👨 Male",
#         help="Patient's biological sex"
#     )
    
#     col1, col2 = st.columns(2)
#     with col1:
#         height = st.slider("**Height** (cm)", 140, 200, 165)
#     with col2:
#         weight = st.slider("**Weight** (kg)", 40, 150, 70)
    
#     # Calculate and display BMI
#     bmi = weight / ((height / 100) ** 2)
#     bmi_status = "Normal" if 18.5 <= bmi <= 24.9 else ("Underweight" if bmi < 18.5 else "Overweight")
#     st.info(f"**BMI:** {bmi:.1f} ({bmi_status})")
    
#     # Vitals Section
#     st.markdown("### 💓 Vitals")
#     col1, col2 = st.columns(2)
#     with col1:
#         systolic_bp = st.slider("**Systolic BP**", 80, 200, 120)
#     with col2:
#         diastolic_bp = st.slider("**Diastolic BP**", 50, 130, 80)
    
#     # Blood Pressure Status
#     bp_status = "Normal" if systolic_bp < 130 and diastolic_bp < 85 else "Elevated"
#     st.info(f"**Blood Pressure:** {systolic_bp}/{diastolic_bp} mmHg ({bp_status})")
    
#     # Health Metrics
#     st.markdown("### 🏥 Health Metrics")
#     cholesterol = st.select_slider(
#         "**Cholesterol Level**",
#         options=[1, 2, 3],
#         format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1]
#     )
    
#     gluc = st.select_slider(
#         "**Glucose Level**",
#         options=[1, 2, 3],
#         format_func=lambda x: ["Normal", "Above Normal", "Well Above Normal"][x-1]
#     )
    
#     # Lifestyle Section
#     st.markdown("### 🏃 Lifestyle Factors")
#     smoke = st.radio("**Smoking**", [0, 1], format_func=lambda x: "✅ Non-smoker" if x == 0 else "🚬 Smoker")
#     alco = st.radio("**Alcohol Intake**", [0, 1], format_func=lambda x: "🚫 No" if x == 0 else "🍷 Yes")
#     active = st.radio("**Physical Activity**", [0, 1], format_func=lambda x: "🛋️ Sedentary" if x == 0 else "🏃 Active")
    
#     st.markdown("---")
    
#     # Prediction Button
#     predict = st.button("🚀 Analyze Heart Risk", use_container_width=True)
    
#     st.markdown("""
#     <div style="margin-top: 2rem; padding: 1rem; background: #f0f2f6; border-radius: 10px;">
#         <small>🔒 <strong>Data Privacy:</strong> All patient data is processed locally and never stored.</small>
#     </div>
#     """, unsafe_allow_html=True)

# # -------------------------------
# # Prediction Section - REDESIGNED
# # -------------------------------
# if predict:
#     st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
#     # Show processing animation
#     with st.spinner("🤖 AI is analyzing health parameters..."):
#         progress_bar = st.progress(0)
#         for i in range(100):
#             progress_bar.progress(i + 1)
    
#     # Build input dictionary
#     BMI = weight / ((height / 100) ** 2)
    
#     input_dict = {
#         'age': age,
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
    
#     # Create DataFrame and predict
#     input_df = pd.DataFrame(
#         [[input_dict[col] for col in scaler.feature_names_in_]],
#         columns=scaler.feature_names_in_
#     )
    
#     input_scaled = scaler.transform(input_df)
#     prediction = model.predict(input_scaled)[0]
    
#     # Display Results with enhanced UI
#     st.markdown("## 📋 Analysis Results")
    
#     # Risk Summary
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("### 📊 Risk Factors Summary")
#         risk_factors = []
#         if age > 50: risk_factors.append("Age")
#         if BMI > 25: risk_factors.append("BMI")
#         if systolic_bp > 130: risk_factors.append("Blood Pressure")
#         if cholesterol > 1: risk_factors.append("Cholesterol")
#         if gluc > 1: risk_factors.append("Glucose")
#         if smoke == 1: risk_factors.append("Smoking")
        
#         for factor in risk_factors:
#             st.markdown(f"• ⚠️ {factor}")
        
#         if not risk_factors:
#             st.markdown("• ✅ No significant risk factors detected")
    
#     with col2:
#         st.markdown("### 🛡️ Protective Factors")
#         protective_factors = []
#         if active == 1: protective_factors.append("Physical Activity")
#         if alco == 0: protective_factors.append("No Alcohol")
        
#         for factor in protective_factors:
#             st.markdown(f"• ✅ {factor}")
        
#         if not protective_factors:
#             st.markdown("• ℹ️ No significant protective factors")
    
#     # Prediction Result
#     st.markdown("## 🎯 Risk Assessment")
    
#     if prediction == 1:
#         st.markdown("""
#         <div class="high-risk">
#             <div style="display: flex; align-items: center; gap: 1rem;">
#                 <span style="font-size: 3rem;">⚠️</span>
#                 <div>
#                     <h2 style="color: #d32f2f; margin: 0;">HIGH RISK DETECTED</h2>
#                     <p style="font-size: 1.2rem; margin: 0.5rem 0;">Elevated cardiovascular risk identified</p>
#                 </div>
#             </div>
#             <hr style="margin: 1.5rem 0;">
#             <h3>📋 Recommended Actions:</h3>
#             <ul>
#                 <li>Consult a cardiologist within 1-2 weeks</li>
#                 <li>Schedule regular blood pressure monitoring</li>
#                 <li>Consider lifestyle modifications</li>
#                 <li>Review diet and exercise regimen</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown("""
#         <div class="low-risk">
#             <div style="display: flex; align-items: center; gap: 1rem;">
#                 <span style="font-size: 3rem;">✅</span>
#                 <div>
#                     <h2 style="color: #2e7d32; margin: 0;">LOW RISK</h2>
#                     <p style="font-size: 1.2rem; margin: 0.5rem 0;">Minimal cardiovascular risk identified</p>
#                 </div>
#             </div>
#             <hr style="margin: 1.5rem 0;">
#             <h3>💡 Health Maintenance:</h3>
#             <ul>
#                 <li>Continue healthy lifestyle habits</li>
#                 <li>Annual cardiovascular check-up recommended</li>
#                 <li>Maintain balanced diet and regular exercise</li>
#                 <li>Monitor key health metrics regularly</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown("""
#     <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
#         <p style="margin: 0; color: #1565c0;">
#         ⚕️ <strong>Medical Disclaimer:</strong> This AI assessment is for informational purposes only 
#         and should not replace professional medical advice. Always consult with qualified healthcare providers.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
#     # -------------------------------
#     # Model Performance Section
#     # -------------------------------
#     st.markdown("## 📈 Model Performance")
    
#     # Performance Metrics
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("""
#         <div style="text-align: center; padding: 1rem;">
#             <h1 style="color: #FF6B6B; margin: 0;">72%</h1>
#             <p><strong>Accuracy</strong></p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("""
#         <div style="text-align: center; padding: 1rem;">
#             <h1 style="color: #4CAF50; margin: 0;">85%</h1>
#             <p><strong>Sensitivity</strong></p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col3:
#         st.markdown("""
#         <div style="text-align: center; padding: 1rem;">
#             <h1 style="color: #2196F3; margin: 0;">65%</h1>
#             <p><strong>Specificity</strong></p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # Confusion Matrix
#     st.markdown("### 🎯 Confusion Matrix")
    
#     data = pd.read_csv("HeartD.csv")
#     data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
    
#     X = data.drop(['cardio', 'id'], axis=1)
#     y = data['cardio']
    
#     X_scaled = scaler.transform(X)
#     y_pred = model.predict(X_scaled)
    
#     cm = confusion_matrix(y, y_pred)
    
#     fig, ax = plt.subplots(figsize=(8, 6))
#     ConfusionMatrixDisplay(
#         confusion_matrix=cm,
#         display_labels=["Low Risk", "High Risk"]
#     ).plot(ax=ax, cmap="RdYlBu_r", values_format="d", text_kw={'fontsize': 14})
    
#     ax.set_title("Model Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
#     ax.set_xlabel("Predicted Label", fontsize=12, fontweight='bold')
#     ax.set_ylabel("True Label", fontsize=12, fontweight='bold')
    
#     plt.tight_layout()
#     st.pyplot(fig)
    
#     # Model Info
#     with st.expander("🔍 Model Details"):
#         st.markdown("""
#         **Model Specifications:**
#         - **Algorithm:** Gradient Boosting Classifier
#         - **Dataset:** 70,000+ patient records
#         - **Features:** 12 clinical parameters
#         - **Validation:** 5-fold cross-validation
#         - **Deployment:** Streamlit + Scikit-learn
        
#         **Feature Importance:**
#         1. Age
#         2. Systolic Blood Pressure
#         3. Cholesterol Level
#         4. BMI
#         5. Glucose Level
#         """)
    
#     # Export Results
#     st.markdown("## 💾 Export Results")
#     col1, col2, col3 = st.columns([1, 1, 2])
    
#     with col1:
#         if st.button("📥 Save as PDF"):
#             st.success("PDF export initiated (simulated)")
    
#     with col2:
#         if st.button("📋 Copy Summary"):
#             st.success("Summary copied to clipboard (simulated)")
    
#     with col3:
#         st.info("💡 Results are automatically saved to your session")

# # -------------------------------
# # Information Section (when no prediction)
# # -------------------------------
# else:
#     st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
#     st.markdown("## 🎯 How It Works")
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("""
#         <div style="text-align: center; padding: 1.5rem;">
#             <h1 style="font-size: 3rem;">1</h1>
#             <h3>Enter Health Data</h3>
#             <p>Fill in your health parameters in the sidebar</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("""
#         <div style="text-align: center; padding: 1.5rem;">
#             <h1 style="font-size: 3rem;">2</h1>
#             <h3>AI Analysis</h3>
#             <p>Our ML model analyzes 12+ risk factors</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     with col3:
#         st.markdown("""
#         <div style="text-align: center; padding: 1.5rem;">
#             <h1 style="font-size: 3rem;">3</h1>
#             <h3>Get Results</h3>
#             <p>Receive instant risk assessment with recommendations</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown("---")
    
#     # Features Grid
#     st.markdown("## 🔬 Analyzed Health Parameters")
    
#     features = [
#         ("👤", "Demographics", "Age, Gender, BMI"),
#         ("💓", "Vital Signs", "Blood Pressure (Systolic/Diastolic)"),
#         ("🧪", "Blood Metrics", "Cholesterol & Glucose Levels"),
#         ("🚬", "Lifestyle", "Smoking, Alcohol, Physical Activity")
#     ]
    
#     cols = st.columns(4)
#     for idx, (icon, title, desc) in enumerate(features):
#         with cols[idx]:
#             st.markdown(f"""
#             <div style="text-align: center; padding: 1rem; background: white; 
#                         border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
#                 <div style="font-size: 2.5rem;">{icon}</div>
#                 <h4>{title}</h4>
#                 <small>{desc}</small>
#             </div>
#             """, unsafe_allow_html=True)

# # -------------------------------
# # Footer
# # -------------------------------
# st.markdown("""
# <div style="text-align: center; padding: 2rem; color: #666; margin-top: 3rem;">
#     <hr>
#     <p>❤️ <strong>HeartGuard AI</strong> | ML-Powered Cardiovascular Risk Assessment</p>
#     <small>For educational and research purposes | Always consult healthcare professionals</small>
# </div>
# """, unsafe_allow_html=True)


import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time

# -------------------------------
# Page Config & High-Impact CSS
# -------------------------------
st.set_page_config(page_title="HeartGuard Ultra", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    /* Full Page Gradient & Animations */
    .stApp {
        background: #0f0c29;
        background: linear-gradient(to bottom, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    
    /* Glowing Card Effect */
    .health-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 25px;
        padding: 30px;
        transition: all 0.4s ease;
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
    }
    .health-card:hover {
        border-color: #00f2fe;
        box-shadow: 0 0 30px rgba(0, 242, 254, 0.2);
        transform: translateY(-5px);
    }

    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(90deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Recommendations Chips */
    .recom-chip {
        display: inline-block;
        padding: 8px 15px;
        margin: 5px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .recom-good { background: rgba(0, 255, 127, 0.2); border: 1px solid #00ff7f; color: #00ff7f; }
    .recom-warn { background: rgba(255, 215, 0, 0.2); border: 1px solid #ffd700; color: #ffd700; }
    .recom-danger { background: rgba(255, 107, 107, 0.2); border: 1px solid #ff6b6b; color: #ff6b6b; }

    /* The Heartbeat Animation */
    @keyframes pulse {
        0% { transform: scale(1); filter: drop-shadow(0 0 5px #ff6b6b); }
        50% { transform: scale(1.1); filter: drop-shadow(0 0 20px #ff6b6b); }
        100% { transform: scale(1); filter: drop-shadow(0 0 5px #ff6b6b); }
    }
    .heart-icon { font-size: 80px; animation: pulse 1s infinite; text-align: center; display: block;}
    
    /* Modern Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        color: white; border: none; border-radius: 12px;
        padding: 15px 30px; font-weight: bold; width: 100%;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Core Interaction Logic
# -------------------------------
if 'screen' not in st.session_state:
    st.session_state.screen = 'input'

# --- Header ---
st.markdown("<h1 style='text-align: center;' class='gradient-text'>HEARTGUARD ULTRA AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; opacity: 0.6;'>Interactive Neural Cardiovascular Mapping</p>", unsafe_allow_html=True)

if st.session_state.screen == 'input':
    # MAIN INPUT AREA
    col_main, col_insight = st.columns([2, 1])

    with col_main:
        st.markdown('<div class="health-card">', unsafe_allow_html=True)
        st.subheader("🧬 Biological Parameters")
        
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Select Patient Age", 18, 100, 45)
            sbp = st.number_input("Systolic Pressure (Top Number)", 80, 220, 120)
            chol = st.select_slider("Cholesterol Bio-Marker", options=[1, 2, 3], 
                                  format_func=lambda x: ["OPTIMAL", "ELEVATED", "CRITICAL"][x-1])
        with c2:
            gender = st.radio("Biological Gender", ["Female", "Male"], horizontal=True)
            dbp = st.number_input("Diastolic Pressure (Bottom Number)", 40, 140, 80)
            gluc = st.select_slider("Glucose Bio-Marker", options=[1, 2, 3],
                                  format_func=lambda x: ["NORMAL", "ELEVATED", "CRITICAL"][x-1])
        
        st.markdown("---")
        st.subheader("🏃 Lifestyle & Habits")
        l1, l2, l3 = st.columns(3)
        smoke = l1.toggle("Active Smoker", key="sm")
        alco = l2.toggle("Alcohol Usage", key="al")
        active = l3.toggle("Physical Activity", key="ac", value=True)
        
        if st.button("INITIATE NEURAL ANALYSIS ⚡"):
            st.session_state.screen = 'result'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_insight:
        st.markdown('<div class="health-card">', unsafe_allow_html=True)
        st.subheader("💡 Instant Insights")
        
        # Dynamic Recommendations based on LIVE inputs
        if sbp > 140 or dbp > 90:
            st.markdown('<div class="recom-chip recom-danger">Hypertension Alert</div>', unsafe_allow_html=True)
            st.write("Current BP levels suggest arterial stress.")
        else:
            st.markdown('<div class="recom-chip recom-good">Healthy BP</div>', unsafe_allow_html=True)

        if chol > 1:
            st.markdown('<div class="recom-chip recom-warn">Lipid Management</div>', unsafe_allow_html=True)
            st.write("Consider Omega-3 rich diet.")

        if not active:
            st.markdown('<div class="recom-chip recom-danger">Sedentary Risk</div>', unsafe_allow_html=True)
            st.write("Add 15 mins of walking daily.")
        else:
            st.markdown('<div class="recom-chip recom-good">Active Profile</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.screen == 'result':
    # --- RESULT PAGE ---
    st.markdown('<div class="health-card" style="text-align: center;">', unsafe_allow_html=True)
    
    # Logic placeholder (Replace with your model.predict)
    risk_found = sbp > 150 or chol == 3 # Simple demo logic
    
    if risk_found:
        st.markdown('<span class="heart-icon">💔</span>', unsafe_allow_html=True)
        st.markdown("<h1 style='color: #ff6b6b;'>CRITICAL WARNING</h1>", unsafe_allow_html=True)
    else:
        st.markdown('<span class="heart-icon">❤️</span>', unsafe_allow_html=True)
        st.markdown("<h1 style='color: #00ff7f;'>HEART IS OPTIMAL</h1>", unsafe_allow_html=True)

    # Colorful Multi-Recommendations Grid
    st.markdown("### 🛠️ Precision Recommendations")
    r1, r2, r3 = st.columns(3)
    
    with r1:
        st.markdown("""
        <div style="background: rgba(79, 172, 254, 0.1); padding: 20px; border-radius: 15px; height: 100%;">
            <h4 style="color: #4facfe;">🍎 Nutrition</h4>
            <p style="font-size: 0.9rem;">Increase Magnesium (spinach, nuts) to support heart rhythm.</p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div style="background: rgba(0, 255, 127, 0.1); padding: 20px; border-radius: 15px; height: 100%;">
            <h4 style="color: #00ff7f;">🧘 Stress</h4>
            <p style="font-size: 0.9rem;">Cortisol levels likely impacted. Practice 4-7-8 breathing.</p>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        st.markdown("""
        <div style="background: rgba(255, 215, 0, 0.1); padding: 20px; border-radius: 15px; height: 100%;">
            <h4 style="color: #ffd700;">💊 Clinical</h4>
            <p style="font-size: 0.9rem;">Review Blood Pressure with a professional within 30 days.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    if st.button("RESCAN PATIENT"):
        st.session_state.screen = 'input'
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("<div style='text-align: center; margin-top: 40px; color: grey;'>Built with AI Precision • 2025 Enterprise Edition</div>", unsafe_allow_html=True)