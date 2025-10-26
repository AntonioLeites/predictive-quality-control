# Prediction API for deployment
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Load trained model + scaler
with open("models/logistic_regression_v1.pkl", "rb") as f:
    model, scaler = pickle.load(f)

app = FastAPI(title="Predictive Quality Control API")

# Define expected input schema
class SensorData(BaseModel):
    oven_temperature_c: float
    molding_pressure_bar: float
    line_speed_mpm: float
    ambient_humidity_pct: float
    material_thickness_mm: float
    material_strength_mpa: float
    cycle_time_sec: float
    machine_vibration_hz: float
    tool_age_hours: float
    shift: int
    operator_experience_years: float
    days_since_maintenance: float

@app.post("/predict")
def predict_defect(data: SensorData, threshold: float = Query(0.5, ge=0.0, le=1.0)):
    """
    Predict defect probability for a single production part.

    - **threshold**: optional float between 0 and 1, default=0.5
    """
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Scale features
    X_scaled = scaler.transform(df)
    
    # Predict probability
    defect_prob = model.predict_proba(X_scaled)[:, 1][0]
    
    # Apply threshold
    defect_prediction = int(defect_prob >= threshold)
    
    return {
        "defect_probability": float(defect_prob),
        "predicted_defect": defect_prediction,
        "threshold_used": threshold
    }

@app.get("/")
def read_root():
    return {"message": "Predictive Quality Control API is running"}
