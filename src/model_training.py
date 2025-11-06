# src/model_training.py
"""
Model training pipeline - Works both locally and in SAP AI Core

Local mode:
  - Reads from: data/synthetic_production_data.csv
  - Saves to: models/logistic_regression_v1.pkl

AI Core mode (ENV AI_CORE_MODE=true):
  - Reads from: /app/data/synthetic_production_data.csv
  - Saves to: /app/model/logistic_regression_v1.pkl
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)

# ✅ Detect execution mode
IS_AI_CORE = os.getenv("AI_CORE_MODE", "").lower() in ["true", "1", "yes"]

if IS_AI_CORE:
    DATA_DIR = "/app/data"
    MODEL_DIR = "/app/model"
    INPUT_FILE = os.path.join(DATA_DIR, "synthetic_production_data.csv")
    OUTPUT_MODEL = os.path.join(MODEL_DIR, "logistic_regression_v1.pkl")
    METRICS_FILE = os.path.join(MODEL_DIR, "metrics.json")
else:
    DATA_DIR = "data"
    MODEL_DIR = "models"
    INPUT_FILE = "data/synthetic_production_data.csv"
    OUTPUT_MODEL = "models/logistic_regression_v1.pkl"
    METRICS_FILE = "models/metrics.json"

def main():
    print("=" * 70)
    print(f"PREDICTIVE QUALITY - MODEL TRAINING")
    print(f"Mode: {'SAP AI Core' if IS_AI_CORE else 'Local Development'}")
    print("=" * 70)
    
    # 1. Load dataset
    print(f"\n📂 Loading data from: {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Training data not found at {INPUT_FILE}\n"
            f"Run 'python src/data_generation.py' first (local mode)"
        )
    
    df = pd.read_csv(INPUT_FILE)
    print(f"✅ Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # 2. Prepare features and target
    X = df.drop(columns=["defective"])
    y = df["defective"]
    
    print(f"\n📊 Dataset split:")
    print(f"   Total samples: {len(df)}")
    print(f"   Defective: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"   Features: {len(X.columns)}")
    
    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # 4. Scale features
    print(f"\n🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train logistic regression
    print(f"\n🚀 Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # Handle imbalanced data
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # 6. Evaluate on test set
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_default = model.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred_default)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n📈 MODEL PERFORMANCE:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_default)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n📋 Confusion Matrix:")
    print(f"   True Negatives:  {tn}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")
    print(f"   True Positives:  {tp}")
    
    # Multi-threshold evaluation
    print(f"\n🎯 Multi-threshold evaluation:")
    thresholds = [0.3, 0.5, 0.7]
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        print(f"\n--- Threshold = {thresh} ---")
        print(classification_report(
            y_test, 
            y_pred_thresh, 
            zero_division=0,
            target_names=['No Defect', 'Defect']
        ))
    
    # 7. Save model + scaler
    print(f"\n💾 Saving model to: {OUTPUT_MODEL}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump((model, scaler), f)
    
    file_size = os.path.getsize(OUTPUT_MODEL) / 1024  # KB
    print(f"✅ Model saved successfully ({file_size:.2f} KB)")
    
    # 8. Save metrics (important for AI Core)
    metrics = {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "samples_train": len(X_train),
        "samples_test": len(X_test),
        "defect_rate": float(y.mean()),
        "features": list(X.columns)
    }
    
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Metrics saved to: {METRICS_FILE}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()