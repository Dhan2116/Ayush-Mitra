"""
Breast Cancer Prediction Module
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🎗️", layout="wide")

st.title("🎗️ Breast Cancer Prediction")
st.markdown("---")

# Load model and scaler
@st.cache_resource
def load_cancer_model():
    try:
        model_path = os.path.join("models", "breast_cancer", "cancer_model.pkl")
        scaler_path = os.path.join("models", "breast_cancer", "scaler.pkl")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please check the integration.")
        return None, None

model, scaler = load_cancer_model()

if model is None:
    st.error("⚠️ Model not loaded. Please check the model files.")
    st.stop()

# Information section
with st.expander("ℹ️ About Breast Cancer Prediction"):
    st.markdown("""
    This model classifies breast tumors as **benign** or **malignant** based on cell nuclei 
    characteristics from the Wisconsin Diagnostic Breast Cancer Dataset.
    
    **Features Used (30 total):**
    
    For each cell nucleus, 10 measurements are computed, each with 3 statistics (mean, SE, worst):
    1. **Radius** - Mean distance from center to perimeter
    2. **Texture** - Standard deviation of gray-scale values
    3. **Perimeter** - Perimeter of the cell nucleus
    4. **Area** - Area of the cell nucleus
    5. **Smoothness** - Local variation in radius lengths
    6. **Compactness** - Perimeter² / area - 1.0
    7. **Concavity** - Severity of concave portions
    8. **Concave Points** - Number of concave portions
    9. **Symmetry** - Symmetry of the cell
    10. **Fractal Dimension** - "Coastline approximation" - 1
    
    **Model Performance:**
    - Algorithm: Logistic Regression
    - Accuracy: 98.25%
    - Excellent precision and recall for both classes
    """)

st.markdown("### Enter Tumor Cell Characteristics")
st.caption("💡 Tip: Use tab navigation for easier data entry")

# Input form with tabs for better organization
with st.form("cancer_form"):
    tab1, tab2, tab3 = st.tabs(["📊 Mean Values", "📈 Standard Error (SE)", "⚠️ Worst Values"])
    
    
    # Tab 1: Mean Values
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            radius_mean = st.number_input("Radius (mean)", 0.0, 50.0, 14.0, 0.1)
            texture_mean = st.number_input("Texture (mean)", 0.0, 50.0, 19.3, 0.1)
            perimeter_mean = st.number_input("Perimeter (mean)", 0.0, 200.0, 92.0, 0.1)
            area_mean = st.number_input("Area (mean)", 0.0, 2500.0, 655.0, 1.0)
            smoothness_mean = st.number_input("Smoothness (mean)", 0.0, 0.5, 0.096, 0.001)
        
        with col2:
            compactness_mean = st.number_input("Compactness (mean)", 0.0, 0.5, 0.104, 0.001)
            concavity_mean = st.number_input("Concavity (mean)", 0.0, 0.5, 0.088, 0.001)
            concave_points_mean = st.number_input("Concave Points (mean)", 0.0, 0.3, 0.048, 0.001)
            symmetry_mean = st.number_input("Symmetry (mean)", 0.0, 0.5, 0.181, 0.001)
            fractal_dimension_mean = st.number_input("Fractal Dimension (mean)", 0.0, 0.1, 0.062, 0.001)
    
    # Tab 2: Standard Error Values
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            radius_se = st.number_input("Radius (SE)", 0.0, 10.0, 0.4, 0.01)
            texture_se = st.number_input("Texture (SE)", 0.0, 10.0, 1.2, 0.01)
            perimeter_se = st.number_input("Perimeter (SE)", 0.0, 30.0, 2.9, 0.01)
            area_se = st.number_input("Area (SE)", 0.0, 500.0, 40.0, 0.1)
            smoothness_se = st.number_input("Smoothness (SE)", 0.0, 0.1, 0.007, 0.0001)
        
        with col2:
            compactness_se = st.number_input("Compactness (SE)", 0.0, 0.2, 0.025, 0.001)
            concavity_se = st.number_input("Concavity (SE)", 0.0, 0.2, 0.031, 0.001)
            concave_points_se = st.number_input("Concave Points (SE)", 0.0, 0.1, 0.011, 0.0001)
            symmetry_se = st.number_input("Symmetry (SE)", 0.0, 0.1, 0.02, 0.001)
            fractal_dimension_se = st.number_input("Fractal Dimension (SE)", 0.0, 0.05, 0.003, 0.0001)
    
    # Tab 3: Worst Values
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            radius_worst = st.number_input("Radius (worst)", 0.0, 50.0, 16.0, 0.1)
            texture_worst = st.number_input("Texture (worst)", 0.0, 50.0, 25.7, 0.1)
            perimeter_worst = st.number_input("Perimeter (worst)", 0.0, 300.0, 107.0, 0.1)
            area_worst = st.number_input("Area (worst)", 0.0, 4000.0, 880.0, 1.0)
            smoothness_worst = st.number_input("Smoothness (worst)", 0.0, 0.5, 0.132, 0.001)
        
        with col2:
            compactness_worst = st.number_input("Compactness (worst)", 0.0, 1.5, 0.254, 0.001)
            concavity_worst = st.number_input("Concavity (worst)", 0.0, 1.5, 0.272, 0.001)
            concave_points_worst = st.number_input("Concave Points (worst)", 0.0, 0.5, 0.114, 0.001)
            symmetry_worst = st.number_input("Symmetry (worst)", 0.0, 0.7, 0.29, 0.001)
            fractal_dimension_worst = st.number_input("Fractal Dimension (worst)", 0.0, 0.3, 0.083, 0.001)
    
    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

# Prediction
if submitted:
    # Prepare data - features must be in exact order used during training
    data = np.array([[
        radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
        compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        radius_se, texture_se, perimeter_se, area_se, smoothness_se,
        compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
        radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
        compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
    ]])
    
    # Scale the features
    with st.spinner("Analyzing tumor characteristics..."):
        data_scaled = scaler.transform(data)
        
        # Make prediction
        prediction = model.predict(data_scaled)[0]
        prob = model.predict_proba(data_scaled)[0]

    st.markdown("---")
    st.markdown("### 📊 Prediction Results")
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 0:  # Malignant (sklearn breast cancer: 0=malignant, 1=benign)
            st.error("### ⚠️ Malignant Tumor Detected")
            st.metric("Malignancy Probability", f"{prob[0]*100:.2f}%")
            st.caption("The tumor characteristics suggest malignancy")
        else:  # Benign
            st.success("### ✅ Benign Tumor Detected")
            st.metric("Benign Probability", f"{prob[1]*100:.2f}%")
            st.caption("The tumor characteristics suggest benign nature")
    
    with col2:
        st.markdown("#### Risk Classification")
        malignant_prob = prob[0] if prediction == 0 else prob[1]
        if malignant_prob < 0.3:
            st.success("🟢 **Low Malignancy Risk**")
            st.caption("Characteristics suggest benign tumor")
        elif malignant_prob < 0.7:
            st.warning("🟡 **Moderate Risk**")
            st.caption("Further diagnostic tests recommended")
        else:
            st.error("🔴 **High Malignancy Risk**")
            st.caption("Immediate medical consultation required")

    # SHAP Explanation
    st.markdown("---")
    st.markdown("### 🔍 Feature Importance Analysis")
    
    with st.spinner("Generating detailed explanation..."):
        try:
            # Feature names
            feature_names = [
                'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
                'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
                'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
            ]
            
            # Create DataFrame for SHAP
            df = pd.DataFrame(data_scaled, columns=feature_names)
            
            explainer = shap.Explainer(model, df)
            shap_values = explainer(df)
            
            # Get SHAP values for the prediction
            shap_val = shap_values[0]
            if shap_val.values.ndim > 1:
                # For benign class (1)
                shap_val = shap_val[:, 1]
            
            # Create waterfall plot
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.plots.waterfall(shap_val, show=False)
            st.pyplot(fig)
            plt.close()
            
            st.info("""
            **How to read this chart:**
            - Red bars push the prediction toward benign
            - Blue bars push the prediction toward malignant
            - Longer bars = stronger influence on the prediction
            - The chart shows how each measurement affects the final diagnosis
            """)
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {str(e)}")

    # Key Findings
    st.markdown("---")
    st.markdown("### 🔬 Key Tumor Characteristics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Radius", f"{radius_mean:.2f}", 
                 help="Average distance from center to  perimeter")
        st.metric("Mean Area", f"{area_mean:.1f}",
                 help="Average cell nucleus area")
    
    with col2:
        st.metric("Mean Concavity", f"{concavity_mean:.3f}",
                 help="Severity of concave portions")
        st.metric("Worst Radius", f"{radius_worst:.2f}",
                 help="Largest radius measurement")
    
    with col3:
        st.metric("Mean Texture", f"{texture_mean:.2f}",
                 help="Variation in gray-scale values")
        st.metric("Worst Area", f"{area_worst:.1f}",
                 help="Largest area measurement")

    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Next Steps & Recommendations")
    
    if prediction == 0:  # Malignant
        st.error("""
        **Malignant Tumor Detected - Immediate Action Required:**
        
        🔸 **Consult an oncologist immediately** - Schedule appointment for comprehensive evaluation
        
        🔸 **Additional diagnostic tests** - Biopsy confirmation, imaging studies (MRI/CT scan)
        
        🔸 **Discuss treatment options** - Surgery, chemotherapy, radiation therapy, targeted therapy
        
        🔸 **Second opinion** - Consider getting evaluation from multiple specialists
        
        🔸 **Support system** - Reach out to cancer support groups and counseling services
        """)
    else:  # Benign
        st.success("""
        **Benign Tumor Detected - Follow-up Recommended:**
        
        🔸 **Regular monitoring** - Schedule follow-up appointments for monitoring
        
        🔸 **Lifestyle modifications** - Maintain healthy diet and regular exercise
        
        🔸 **Self-examination** - Continue monthly breast self-examinations
        
        🔸 **Annual screenings** - Continue regular mammography as per guidelines
        
        🔸 **Stay informed** - Be aware of any changes and report to healthcare provider
        """)
    
    st.warning("""
    ⚠️ **Critical Medical Disclaimer:** 
    
    This AI-based prediction is for **informational and educational purposes ONLY**. It should NOT 
    be used as the sole basis for medical decisions.
    
    - **Not a substitute for professional diagnosis** - Always consult qualified oncologists
    - **Biopsy required for confirmation** - Definitive diagnosis requires tissue examination
    - **False positives/negatives possible** - No AI system is 100% accurate
    - **Individual variation** - Every case is unique and requires personalized evaluation
    
    **Seek immediate medical attention** from qualified healthcare providers for proper 
    diagnosis, treatment planning, and ongoing care.
    """)
