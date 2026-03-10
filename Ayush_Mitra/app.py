"""
Ayush Mitra - Centralized Medical Diagnosis Helper
A comprehensive system for multiple disease predictions
"""

import streamlit as st

st.set_page_config(
    page_title="Ayush Mitra - Medical Diagnosis Helper",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Ayush Mitra</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Medical Diagnosis Assistant</p>', unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
## Welcome to Ayush Mitra

Ayush Mitra is a comprehensive medical diagnosis helper system that leverages machine learning 
to provide risk assessments for multiple conditions.

### Available Prediction Models:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-box">
        <h3>Diabetes Prediction</h3>
        <p>Predict diabetes risk based on clinical parameters including glucose levels, BMI, 
        blood pressure, and family history.</p>
        <p><strong>Model:</strong> Random Forest Classifier</p>
        <p><strong>Features:</strong> 8 clinical parameters</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <h3>Heart Disease Prediction</h3>
        <p>Assess cardiovascular disease risk using key health indicators including cholesterol, 
        blood pressure, and ECG results.</p>
        <p><strong>Model:</strong> Random Forest Classifier</p>
        <p><strong>Features:</strong> 13 cardiac indicators</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-box">
        <h3>Breast Cancer Prediction</h3>
        <p>Classify breast tumors as benign or malignant using cell nuclei characteristics 
        from diagnostic imaging.</p>
        <p><strong>Model:</strong> Logistic Regression</p>
        <p><strong>Features:</strong> 30 cell nuclei measurements</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
## How to Use

1. **Select a prediction model** from the sidebar menu
2. **Enter the required patient information** in the input form
3. **Click 'Predict'** to get the risk assessment
4. **Review the results** including probability scores and SHAP explanations

### Important Disclaimer

This system is designed for educational and research purposes only. It should NOT be used as a 
substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of 
qualified health providers with any questions regarding medical conditions.

### Privacy & Security

All predictions are performed locally. No patient data is stored or transmitted to external servers.
""")

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #888; padding: 20px;'>
    <p>Ayush Mitra v1.0 | Powered by Machine Learning & Streamlit</p>
    <p>© 2026 - For Educational Purposes Only</p>
</div>
""", unsafe_allow_html=True)
