# Prediction API for deployment
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI(title="Predictive Quality Control API")

with open("models/logistic_regression_v1.pkl", "rb") as f:
    model, scaler = pickle.load(f)

@app.post("/predict")
def predict_quality(features: dict):
    X = np.array([list(features.values())]).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0, 1]
    return {"defect_probability": round(float(prob), 3)}
