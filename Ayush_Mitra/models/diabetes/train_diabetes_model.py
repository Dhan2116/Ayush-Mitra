"""
Training script for Diabetes Prediction Model
Ayush Mitra - Diabetes Module
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

from preprocess_diabetes import preprocess_pipeline

# Configuration
MODEL_OUTPUT_PATH = "diabetes_model.pkl"
ARTIFACT_DIR = "artifacts"

def train_and_evaluate_models():
    """Train multiple models and compare performance"""
    
    print("="*60)
    print("AYUSH MITRA - Diabetes Model Training")
    print("="*60)
    
    # Preprocess data
    print("\n📊 Running preprocessing pipeline...")
    X_train, X_test, y_train, y_test = preprocess_pipeline()
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    print("\n🤖 Training models...")
    print("-" * 60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
        
        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "AUC-ROC": round(auc, 4)
        })
        
        trained_models[name] = model
        
        print(f"✅ {name} - Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    
    # Display comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Select best model (based on F1 score)
    best_idx = results_df['F1 Score'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_model = trained_models[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print("="*60)
    
    # Detailed evaluation of best model
    y_pred_best = best_model.predict(X_test)
    
    print("\n📈 Detailed Classification Report:")
    print(classification_report(y_test, y_pred_best, 
                                target_names=['Non-Diabetic', 'Diabetic']))
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best)
    print(cm)
    print(f"\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    # Save best model
    joblib.dump(best_model, MODEL_OUTPUT_PATH)
    print(f"\n💾 Best model saved to: {MODEL_OUTPUT_PATH}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    generate_visualizations(results_df, best_model, X_test, y_test)
    
    return best_model, results_df

def generate_visualizations(results_df, best_model, X_test, y_test):
    """Generate and save performance visualizations"""
    
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    # 1. Model Comparison Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        results_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False, color='skyblue')
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {ARTIFACT_DIR}/model_comparison.png")
    plt.close()
    
    # 2. Confusion Matrix Heatmap
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Diabetic', 'Diabetic'],
                yticklabels=['Non-Diabetic', 'Diabetic'])
    plt.title('Confusion Matrix - Best Model', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {ARTIFACT_DIR}/confusion_matrix.png")
    plt.close()
    
    # 3. ROC Curve (if model supports probability)
    if hasattr(best_model, 'predict_proba'):
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Best Model', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {ARTIFACT_DIR}/roc_curve.png")
        plt.close()
    
    # 4. Feature Importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], color='steelblue')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance - Best Model', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACT_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {ARTIFACT_DIR}/feature_importance.png")
        plt.close()

if __name__ == "__main__":
    try:
        best_model, results = train_and_evaluate_models()
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nModel ready for deployment in Ayush Mitra system")
        print(f"Model file: {MODEL_OUTPUT_PATH}")
        print(f"Visualizations: {ARTIFACT_DIR}/")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
