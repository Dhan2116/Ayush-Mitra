# Ayush Mitra - AI-Powered Medical Diagnosis System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

Ayush Mitra is a comprehensive web-based medical diagnosis assistant that leverages machine learning to provide risk assessments for multiple health conditions. The system integrates three specialized prediction models for diabetes, heart disease, and breast cancer detection, offering healthcare professionals and individuals valuable insights through an intuitive interface.

## Features

### 1. Diabetes Prediction
- **Algorithm:** Random Forest Classifier
- **Accuracy:** ~85%
- **Input Features:** 8 clinical parameters
  - Pregnancies, Glucose, Blood Pressure, Skin Thickness
  - Insulin, BMI, Diabetes Pedigree Function, Age
- **Dataset:** Pima Indians Diabetes Database

### 2. Heart Disease Prediction
- **Algorithm:** Random Forest Classifier
- **Input Features:** 13 cardiac indicators
  - Age, Sex, Chest Pain Type, Resting Blood Pressure
  - Cholesterol, Fasting Blood Sugar, ECG Results
  - Max Heart Rate, Exercise Induced Angina, ST Depression
  - Slope, Number of Major Vessels, Thalassemia
- **Dataset:** Cleveland Heart Disease Database

### 3. Breast Cancer Prediction
- **Algorithm:** Logistic Regression
- **Accuracy:** 98.25%
- **Input Features:** 30 tumor characteristics
  - Radius, Texture, Perimeter, Area, Smoothness
  - Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension
  - (Mean, Standard Error, and Worst values for each)
- **Dataset:** Wisconsin Breast Cancer Dataset

## Key Capabilities

- **Model Explainability:** SHAP (SHapley Additive exPlanations) integration for transparent predictions
- **Interactive Visualizations:** Real-time charts and feature importance displays
- **Medical Recommendations:** Actionable health advice based on prediction results
- **Responsive Design:** Clean, user-friendly interface built with Streamlit
- **Data Preprocessing:** Automated handling with SMOTE for class imbalance

## Technology Stack

- **Frontend:** Streamlit
- **Machine Learning:** scikit-learn, imbalanced-learn
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Data Processing:** Pandas, NumPy
- **Model Persistence:** Joblib

## Project Structure

```
Ayush_Mitra/
│
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
│
├── pages/                          # Streamlit multi-page app
│   ├── 1_Diabetes_Prediction.py
│   ├── 2_Heart_Disease_Prediction.py
│   └── 3_Breast_Cancer_Prediction.py
│
├── models/                         # ML models and training scripts
│   ├── diabetes/
│   │   ├── train_diabetes_model.py
│   │   ├── preprocess_diabetes.py
│   │   ├── diabetes_model.pkl      # (gitignored)
│   │   └── diabetes_data.csv       # (gitignored)
│   │
│   ├── heart_disease/
│   │   ├── heart_model.pkl         # (gitignored)
│   │   ├── scaler.pkl              # (gitignored)
│   │   └── heart_data.csv          # (gitignored)
│   │
│   └── breast_cancer/
│       ├── train_cancer_model.py
│       ├── cancer_model.pkl        # (gitignored)
│       └── scaler.pkl              # (gitignored)
│
├── utils/                          # Utility functions
│   └── helpers.py
│
└── .streamlit/                     # Streamlit configuration
    └── config.toml
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Ayush_Mitra
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare datasets**
   
   Place your dataset files in the respective model directories:
   - `models/diabetes/diabetes_data.csv`
   - `models/heart_disease/heart_data.csv`
   - Breast cancer uses sklearn built-in dataset

5. **Train models**
   ```bash
   # Diabetes model
   cd models/diabetes
   python train_diabetes_model.py
   
   # Breast cancer model
   cd ../breast_cancer
   python train_cancer_model.py
   cd ../..
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

7. **Access the application**
   - Open browser to `http://localhost:8501`

## Usage

### Making Predictions

1. **Select Model:** Choose from the sidebar menu
   - Diabetes Prediction
   - Heart Disease Prediction
   - Breast Cancer Prediction

2. **Enter Patient Data:** Fill in the required clinical parameters

3. **Generate Prediction:** Click the "Predict" button

4. **Review Results:**
   - Risk assessment (High Risk/Low Risk or Malignant/Benign)
   - Prediction probability
   - SHAP explanation visualizations
   - Medical recommendations

### Example Test Cases

#### Diabetes (High Risk)
```
Pregnancies: 6
Glucose: 148
Blood Pressure: 72
Skin Thickness: 35
Insulin: 0
BMI: 33.6
Diabetes Pedigree Function: 0.627
Age: 50
```

#### Heart Disease (High Risk)
```
Age: 63
Sex: Male
Chest Pain Type: Asymptomatic
Resting BP: 145
Cholesterol: 233
Fasting Blood Sugar: >120 mg/dl
Rest ECG: ST-T wave abnormality
Max Heart Rate: 150
Exercise Angina: Yes
ST Depression: 2.3
Slope: Downsloping
Major Vessels: 0
Thalassemia: Fixed defect
```

## Model Training

Each model directory contains training scripts that:
- Load and preprocess data
- Handle class imbalance (SMOTE for diabetes)
- Train multiple algorithms
- Compare performance metrics
- Save the best performing model

### Training Commands
```bash
# Diabetes
python models/diabetes/train_diabetes_model.py

# Breast Cancer
python models/breast_cancer/train_cancer_model.py
```

## Model Performance

| Model | Algorithm | Accuracy | Precision | Recall |
|-------|-----------|----------|-----------|--------|
| Diabetes | Random Forest | ~85% | 0.78 | 0.66 |
| Heart Disease | Random Forest | - | - | - |
| Breast Cancer | Logistic Regression | 98.25% | 0.98 | 0.99 |

## Dependencies

Core requirements:
- streamlit >= 1.28.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- shap >= 0.42.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- joblib >= 1.3.0
- imbalanced-learn >= 0.11.0

See `requirements.txt` for complete list.

## Medical Disclaimer

**IMPORTANT:** This system is designed for educational and informational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for medical decisions.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## Future Enhancements

- [ ] User authentication and session management
- [ ] Prediction history tracking
- [ ] PDF report generation
- [ ] Mobile-responsive design improvements
- [ ] Additional disease models (lung cancer, kidney disease, etc.)
- [ ] RESTful API for model predictions
- [ ] Cloud deployment (AWS, Azure, Streamlit Cloud)
- [ ] Model retraining pipeline with new data

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

**Datasets:**
- Pima Indians Diabetes Database (UCI Machine Learning Repository)
- Cleveland Heart Disease Database
- Wisconsin Breast Cancer Dataset (sklearn.datasets)

**Libraries:** scikit-learn, Streamlit, SHAP contributors

## Contact

For questions, suggestions, or collaborations:
- Create an issue in this repository
- Pull requests are welcome

## Version History

- **v1.0.0** (March 2026) - Initial release
  - Diabetes prediction model
  - Heart disease prediction model
  - Breast cancer prediction model
  - SHAP explainability integration
  - Multi-page Streamlit interface

---

**Built with care for better healthcare accessibility**
