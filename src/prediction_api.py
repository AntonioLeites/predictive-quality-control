# src/prediction_api.py
"""
Prediction API - Works both locally and in SAP AI Core

Local mode (python -m uvicorn src.prediction_api:app --reload):
  - Model from: models/logistic_regression_v1.pkl
  - Port: 8000

AI Core mode (ENV AI_CORE_MODE=true):
  - Model from: /app/model/logistic_regression_v1.pkl
  - Port: 9001
"""
import os
import pickle
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

# ✅ Detect execution mode
IS_AI_CORE = os.getenv("AI_CORE_MODE", "").lower() in ["true", "1", "yes"]

if IS_AI_CORE:
    MODEL_FILE = "/app/model/logistic_regression_v1.pkl"
    PORT = 9001
else:
    MODEL_FILE = "models/logistic_regression_v1.pkl"
    PORT = 8000

# FastAPI app
app = FastAPI(
    title="Predictive Quality Control API",
    version="1.0.0",
    description="Real-time defect prediction for manufacturing (Logistic Regression)"
)

# Global model variable
model = None
scaler = None

@app.on_event("startup")
async def load_model():
    """Load model and scaler at startup"""
    global model, scaler
    
    print("=" * 70)
    print(f"PREDICTIVE QUALITY - SERVING API")
    print(f"Mode: {'SAP AI Core' if IS_AI_CORE else 'Local Development'}")
    print("=" * 70)
    
    # Check for STORAGE_URI (AI Core specific)
    if IS_AI_CORE:
        storage_uri = os.getenv("STORAGE_URI", "Not set")
        print(f"📦 STORAGE_URI: {storage_uri}")
    
    print(f"📂 Loading model from: {MODEL_FILE}")
    
    if not os.path.exists(MODEL_FILE):
        error_msg = f"Model file not found at {MODEL_FILE}"
        if not IS_AI_CORE:
            error_msg += "\nRun 'python src/model_training.py' first"
        print(f"❌ {error_msg}")
        raise FileNotFoundError(error_msg)
    
    # Load model + scaler
    with open(MODEL_FILE, 'rb') as f:
        model, scaler = pickle.load(f)
    
    print(f"✅ Model loaded successfully")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Scaler type: {type(scaler).__name__}")
    print(f"   Serving on port: {PORT}")
    print("=" * 70)

class PredictionRequest(BaseModel):
    oven_temperature_c: float = Field(..., description="Oven temperature (°C)", example=240.0)
    molding_pressure_bar: float = Field(..., description="Molding pressure (bar)", example=160.0)
    line_speed_mpm: float = Field(..., description="Line speed (m/min)", example=46.0)
    ambient_humidity_pct: float = Field(..., description="Ambient humidity (%)", example=40.0)
    material_thickness_mm: float = Field(..., description="Material thickness (mm)", example=2.5)
    material_strength_mpa: float = Field(..., description="Material strength (MPa)", example=355.0)
    cycle_time_sec: float = Field(..., description="Cycle time (sec)", example=12.0)
    machine_vibration_hz: float = Field(..., description="Machine vibration (Hz)", example=1.5)
    tool_age_hours: float = Field(..., description="Tool age (hours)", example=420.0)
    shift: int = Field(..., description="Shift (1, 2, or 3)", example=2)
    operator_experience_years: float = Field(..., description="Operator experience (years)", example=3.0)
    days_since_maintenance: float = Field(..., description="Days since maintenance", example=15.0)

class PredictionResponse(BaseModel):
    defect_predicted: bool
    defect_probability: float
    risk_level: str
    threshold_used: float
    model_version: str = "1.0.0"

@app.get("/")
def root():
    """API information"""
    return {
        "service": "Predictive Quality Control",
        "version": "1.0.0",
        "status": "running",
        "mode": "AI Core" if IS_AI_CORE else "Local",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint (required by SAP AI Core)"""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": type(model).__name__,
        "scaler_type": type(scaler).__name__
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Decision threshold")
):
    """
    Predict defect probability for a production part
    
    Args:
        request: Production parameters from sensors
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        Prediction with probability and risk level
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert to DataFrame (IMPORTANT: column names must match training data)
    data = pd.DataFrame([{
        'oven_temperature_c': request.oven_temperature_c,
        'molding_pressure_bar': request.molding_pressure_bar,
        'line_speed_mpm': request.line_speed_mpm,
        'ambient_humidity_pct': request.ambient_humidity_pct,
        'material_thickness_mm': request.material_thickness_mm,
        'material_strength_mpa': request.material_strength_mpa,
        'cycle_time_sec': request.cycle_time_sec,
        'machine_vibration_hz': request.machine_vibration_hz,
        'tool_age_hours': request.tool_age_hours,
        'shift': request.shift,
        'operator_experience_years': request.operator_experience_years,
        'days_since_maintenance': request.days_since_maintenance
    }])
    
    # Scale and predict
    data_scaled = scaler.transform(data)
    probability = float(model.predict_proba(data_scaled)[0, 1])
    prediction = probability >= threshold
    
    # Risk level classification
    if probability >= 0.7:
        risk_level = "HIGH"
    elif probability >= 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return PredictionResponse(
        defect_predicted=prediction,
        defect_probability=probability,
        risk_level=risk_level,
        threshold_used=threshold
    )

@app.get("/model/info")
def model_info():
    """Get model metadata"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = {
        "model_type": "Logistic Regression",
        "scaler_type": type(scaler).__name__,
        "features_expected": 12,
        "feature_names": [
            "oven_temperature_c", "molding_pressure_bar", "line_speed_mpm",
            "ambient_humidity_pct", "material_thickness_mm", "material_strength_mpa",
            "cycle_time_sec", "machine_vibration_hz", "tool_age_hours",
            "shift", "operator_experience_years", "days_since_maintenance"
        ]
    }
    
    # Add coefficients if available
    if hasattr(model, 'coef_'):
        info["coefficients"] = model.coef_[0].tolist()
        info["intercept"] = float(model.intercept_[0])
    
    return info

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)