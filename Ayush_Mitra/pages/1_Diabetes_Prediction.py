"""
Diabetes Prediction Module
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩸", layout="wide")

st.title("Diabetes Prediction")
st.markdown("---")

@st.cache_resource
def load_diabetes_model():
    try:
        model_path = os.path.join("models", "diabetes", "diabetes_model.pkl")
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please ensure the diabetes model is trained and saved.")
        return None

model = load_diabetes_model()

if model is None:
    st.info("""
    ### Setup Required
    1. Navigate to `models/diabetes/` directory
    2. Run `python train_diabetes_model.py` to train the model
    3. Refresh this page
    """)
    st.stop()

with st.expander("About Diabetes Prediction"):
    st.markdown("""
    This model predicts the likelihood of diabetes based on diagnostic measurements.
    
    **Features Used:**
    - **Pregnancies:** Number of times pregnant
    - **Glucose:** Plasma glucose concentration (2 hours in an oral glucose tolerance test)
    - **Blood Pressure:** Diastolic blood pressure (mm Hg)
    - **Skin Thickness:** Triceps skin fold thickness (mm)
    - **Insulin:** 2-Hour serum insulin (mu U/ml)
    - **BMI:** Body mass index (weight in kg/(height in m)^2)
    - **Diabetes Pedigree Function:** Diabetes pedigree function (genetic factor)
    - **Age:** Age in years
    
    **Model Performance:**
    - Algorithm: Random Forest Classifier
    - Provides probability scores and feature importance explanations
    """)

st.markdown("### Enter Patient Information")

with st.form("diabetes_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, 
                                     help="Number of times pregnant")
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120,
                                 help="Plasma glucose concentration")
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70,
                                        help="Diastolic blood pressure")
    
    with col2:
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20,
                                        help="Triceps skin fold thickness")
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=900, value=80,
                                 help="2-Hour serum insulin")
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1,
                            help="Body mass index")
    
    with col3:
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, 
                             value=0.5, step=0.01,
                             help="Genetic diabetes likelihood")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=33,
                             help="Age in years")

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                     insulin, bmi, dpf, age]])
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    df = pd.DataFrame(data, columns=feature_names)

    with st.spinner("Analyzing..."):
        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0]

    st.markdown("---")
    st.markdown("### Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("### High Risk of Diabetes")
            st.metric("Diabetes Probability", f"{prob[1]*100:.2f}%")
        else:
            st.success("### Low Risk of Diabetes")
            st.metric("Non-Diabetes Probability", f"{prob[0]*100:.2f}%")
    
    with col2:
        st.markdown("#### Risk Level")
        if prob[1] < 0.3:
            st.info("**Low Risk** - Continue healthy lifestyle")
        elif prob[1] < 0.7:
            st.warning("**Moderate Risk** - Consider lifestyle modifications")
        else:
            st.error("**High Risk** - Consult healthcare provider")

    st.markdown("---")
    st.markdown("### Feature Importance Analysis")
    
    with st.spinner("Generating explanation..."):
        try:
            explainer = shap.Explainer(model, df)
            shap_values = explainer(df)
            
            shap_val = shap_values[0]
            if shap_val.values.ndim > 1:
                shap_val = shap_val[:, 1]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_val, show=False)
            st.pyplot(fig)
            plt.close()
            
            st.info("""
            **How to read this chart:**
            - Red bars push the prediction toward diabetes
            - Blue bars push the prediction away from diabetes
            - Longer bars indicate stronger influence
            """)
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {str(e)}")

    st.markdown("---")
    st.markdown("### Recommendations")
    
    recommendations = []
    
    if glucose > 140:
        recommendations.append("**High glucose levels detected** - Monitor blood sugar regularly")
    if bmi > 30:
        recommendations.append("**BMI indicates obesity** - Consider weight management program")
    if blood_pressure > 90:
        recommendations.append("**Elevated blood pressure** - Monitor and manage hypertension")
    if age > 45:
        recommendations.append("**Age is a risk factor** - Regular health screenings recommended")
    if dpf > 0.8:
        recommendations.append("**Strong family history** - Extra vigilance recommended")
    
    if recommendations:
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.success("All parameters within normal ranges. Continue healthy lifestyle!")
    
    st.warning("""
    **Medical Disclaimer:** This prediction is for informational purposes only and should not 
    replace professional medical advice. Please consult with a healthcare provider for proper 
    diagnosis and treatment.
    """)
