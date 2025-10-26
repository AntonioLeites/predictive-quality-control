# Model training pipeline with multi-threshold evaluation
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Load dataset
df = pd.read_csv("data/synthetic_production_data.csv")

# Features and target
X = df.drop(columns=["defective"])
y = df["defective"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Probabilities for test set
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.3f}\n")

# Evaluate multiple thresholds
thresholds = [0.3, 0.5, 0.7]

for thresh in thresholds:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    print(f"Classification Report (Threshold={thresh}):")
    print(classification_report(y_test, y_pred_thresh, zero_division=0))

# Save model + scaler
with open("models/logistic_regression_v1.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("\nModel trained and saved to models/logistic_regression_v1.pkl")
