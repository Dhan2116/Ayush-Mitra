"""
Heart Disease Prediction Module
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")

st.title("❤️ Heart Disease Prediction")
st.markdown("---")

# Load model and scaler
@st.cache_resource
def load_heart_model():
    try:
        model_path = os.path.join("models", "heart_disease", "heart_model.pkl")
        scaler_path = os.path.join("models", "heart_disease", "scaler.pkl")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please check the integration.")
        return None, None

model, scaler = load_heart_model()

if model is None:
    st.error("⚠️ Model not loaded. Please check the model files.")
    st.stop()

# Information section
with st.expander("ℹ️ About Heart Disease Prediction"):
    st.markdown("""
    This model predicts the likelihood of heart disease based on various cardiac indicators 
    from the Cleveland Heart Disease Database.
    
    **Features Used:**
    - **Age:** Age in years
    - **Sex:** Gender (1 = male, 0 = female)
    - **CP (Chest Pain Type):** 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic
    - **Trestbps:** Resting blood pressure (mm Hg)
    - **Chol:** Serum cholesterol (mg/dl)
    - **FBS:** Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    - **Restecg:** Resting ECG results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy)
    - **Thalach:** Maximum heart rate achieved
    - **Exang:** Exercise induced angina (1 = yes, 0 = no)
    - **Oldpeak:** ST depression induced by exercise relative to rest
    - **Slope:** Slope of peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping)
    - **CA:** Number of major vessels colored by fluoroscopy (0-3)
    - **Thal:** Thalassemia (0 = normal, 1 = fixed defect, 2 = reversible defect)
    
    **Model Performance:**
    - Algorithm: Random Forest Classifier
    - High accuracy for cardiovascular risk assessment
    """)

st.markdown("### Enter Patient Information")

# Input form
with st.form("heart_form"):
    col1, col2, col3, col4 = st.columns(4)
    
    
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=50,
                             help="Patient's age in years")
        sex = st.selectbox("Sex", options=["Female", "Male"],
                          help="Patient's biological sex")
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3],
                         format_func=lambda x: ["Typical Angina", "Atypical Angina", 
                                               "Non-Anginal Pain", "Asymptomatic"][x],
                         help="Type of chest pain experienced")
    
    with col2:
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 
                                   min_value=80, max_value=200, value=120,
                                   help="Resting blood pressure in mm Hg")
        chol = st.number_input("Cholesterol (mg/dl)", 
                              min_value=100, max_value=600, value=200,
                              help="Serum cholesterol in mg/dl")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                          options=["No", "Yes"],
                          help="Is fasting blood sugar > 120 mg/dl?")
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2],
                               format_func=lambda x: ["Normal", "ST-T Wave Abnormality", 
                                                     "Left Ventricular Hypertrophy"][x],
                               help="Resting electrocardiographic results")
    
    with col3:
        thalach = st.number_input("Maximum Heart Rate", 
                                 min_value=60, max_value=220, value=150,
                                 help="Maximum heart rate achieved")
        exang = st.selectbox("Exercise Induced Angina", 
                            options=["No", "Yes"],
                            help="Exercise induced angina?")
        oldpeak = st.number_input("ST Depression", 
                                 min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                                 help="ST depression induced by exercise relative to rest")
    
    with col4:
        slope = st.selectbox("Slope of Peak Exercise ST Segment", options=[0, 1, 2],
                            format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x],
                            help="Slope of the peak exercise ST segment")
        ca = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3],
                         help="Number of major vessels colored by fluoroscopy")
        thal = st.selectbox("Thalassemia", options=[0, 1, 2],
                           format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x],
                           help="Thalassemia test result")
    
    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

# Prediction
if submitted:
    # Convert categorical inputs
    sex_val = 1 if sex == "Male" else 0
    fbs_val = 1 if fbs == "Yes" else 0
    exang_val = 1 if exang == "Yes" else 0
    
    # Prepare data (features must match training order)
    # Order: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    data = np.array([[age, sex_val, cp, trestbps, chol, fbs_val, 
                     restecg, thalach, exang_val, oldpeak, slope, ca, thal]])
    
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                     'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Scale the features
    with st.spinner("Analyzing..."):
        data_scaled = scaler.transform(data)
        
        # Make prediction
        prediction = model.predict(data_scaled)[0]
        prob = model.predict_proba(data_scaled)[0]

    st.markdown("---")
    st.markdown("### 📊 Prediction Results")
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("### ❤️‍🩹 Heart Disease Risk Detected")
            st.metric("Disease Probability", f"{prob[1]*100:.2f}%")
        else:
            st.success("### ✅ Low Risk of Heart Disease")
            st.metric("Healthy Probability", f"{prob[0]*100:.2f}%")
    
    with col2:
        st.markdown("#### Risk Level")
        if prob[1] < 0.3:
            st.info("🟢 **Low Risk** - Continue healthy lifestyle")
        elif prob[1] < 0.7:
            st.warning("🟡 **Moderate Risk** - Consider lifestyle modifications and regular checkups")
        else:
            st.error("🔴 **High Risk** - Consult cardiologist immediately")

    # SHAP Explanation
    st.markdown("---")
    st.markdown("### 🔍 Feature Importance Analysis")
    
    with st.spinner("Generating explanation..."):
        try:
            # Create DataFrame for SHAP
            df = pd.DataFrame(data_scaled, columns=feature_names)
            
            explainer = shap.Explainer(model, df)
            shap_values = explainer(df)
            
            # Get SHAP values for the prediction
            shap_val = shap_values[0]
            if shap_val.values.ndim > 1:
                shap_val = shap_val[:, 1]  # For positive class (heart disease)
            
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_val, show=False)
            st.pyplot(fig)
            plt.close()
            
            st.info("""
            **How to read this chart:**
            - Red bars push the prediction toward heart disease
            - Blue bars push the prediction away from heart disease
            - Longer bars indicate stronger influence on the prediction
            """)
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {str(e)}")

    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Recommendations")
    
    recommendations = []
    
    if chol > 240:
        recommendations.append("🔸 **High cholesterol** - Consider dietary changes and consult with doctor")
    if trestbps > 140:
        recommendations.append("🔸 **Elevated blood pressure** - Monitor regularly and manage hypertension")
    if thalach > 180:
        recommendations.append("🔸 **High maximum heart rate** - Discuss with cardiologist")
    if age > 55:
        recommendations.append("🔸 **Age factor** - Regular cardiac screenings recommended")
    if cp > 0:
        recommendations.append("🔸 **Chest pain reported** - Seek medical evaluation")
    if exang == "Yes":
        recommendations.append("🔸 **Exercise-induced angina** - Cardiac stress test recommended")
    if fbs == "Yes":
        recommendations.append("🔸 **Elevated fasting blood sugar** - Screen for diabetes")
    
    if recommendations:
        for rec in recommendations:
            st.markdown(rec)
    else:
        st.success("✅ Major cardiac indicators within normal ranges. Continue healthy lifestyle!")
    
    st.warning("""
    ⚠️ **Medical Disclaimer:** This prediction is for informational purposes only and should not 
    replace professional medical advice. Please consult with a qualified cardiologist or healthcare 
    provider for proper diagnosis and treatment.
    """)
