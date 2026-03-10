"""
Preprocessing pipeline for Pima Indians Diabetes Dataset
Ayush Mitra - Diabetes Module
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# ---------------------------- CONFIG ----------------------------
DATA_PATH = "diabetes_data.csv"
TARGET_COL = "Outcome"
ZERO_AS_NAN_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
TEST_SIZE = 0.2
RANDOM_STATE = 42
APPLY_SMOTE = True
ARTIFACT_DIR = "artifacts"
# ----------------------------------------------------------------

def load_data(path=DATA_PATH):
    """Load the diabetes dataset"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape}")
    return df

def mark_zeros_as_nan(df, cols=ZERO_AS_NAN_COLS):
    """Replace zero values with NaN for specified columns"""
    df = df.copy()
    for col in cols:
        df[col] = df[col].replace(0, np.nan)
    print(f"Zeros replaced with NaN in: {cols}")
    return df

def impute_missing_values(df, cols=ZERO_AS_NAN_COLS, strategy="median"):
    """Impute missing values using specified strategy"""
    df = df.copy()
    for col in cols:
        if strategy == "median":
            df[col].fillna(df[col].median(), inplace=True)
        elif strategy == "mean":
            df[col].fillna(df[col].mean(), inplace=True)
    print(f"Missing values imputed using {strategy}")
    return df

def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def apply_smote_resampling(X_train, y_train):
    """Apply SMOTE to balance the dataset"""
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied. New training size: {X_train_resampled.shape}")
    return X_train_resampled, y_train_resampled

def preprocess_pipeline(data_path=DATA_PATH, apply_smote=APPLY_SMOTE):
    """
    Complete preprocessing pipeline
    
    Returns:
        X_train, X_test, y_train, y_test (scaled and preprocessed)
    """
    # Load data
    df = load_data(data_path)
    
    # Handle missing values (zeros)
    df = mark_zeros_as_nan(df)
    df = impute_missing_values(df)
    
    # Split features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train-test split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Apply SMOTE if enabled
    if apply_smote:
        X_train_scaled, y_train = apply_smote_resampling(X_train_scaled, y_train)
    
    # Save artifacts
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    np.save(os.path.join(ARTIFACT_DIR, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(ARTIFACT_DIR, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(ARTIFACT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(ARTIFACT_DIR, 'y_test.npy'), y_test)
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, 'scaler.joblib'))
    
    print(f"Artifacts saved to {ARTIFACT_DIR}/")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    # Test preprocessing
    try:
        X_train, X_test, y_train, y_test = preprocess_pipeline()
        print("\n✅ Preprocessing completed successfully!")
        print(f"Final shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure 'diabetes_data.csv' is in the current directory")
