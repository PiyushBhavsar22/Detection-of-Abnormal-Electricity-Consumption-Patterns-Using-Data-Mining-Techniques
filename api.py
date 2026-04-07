from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model at module level
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "theft_detection_model.pkl")
features_path = os.path.join(base_dir, "model_features.pkl")

try:
    model = joblib.load(model_path)
    model_features = joblib.load(features_path)
    logger.info(f"Model loaded successfully with {len(model_features)} features")
except FileNotFoundError as e:
    logger.warning(f"Model files not found: {e}. API will fail on predictions.")
    model = None
    model_features = None

app = FastAPI(title="Electricity Theft Detection API")

# Add CORS middleware for dashboard browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Update the Validator to match the actual feature names
class ConsumerBehavior(BaseModel):
    total_daily_kwh: float = Field(..., ge=0)
    peak_to_offpeak_ratio: float = Field(..., ge=0)
    daily_variance: float = Field(..., ge=0) # Renamed from variance_kwh
    temperature_celsius: float = Field(..., ge=-50, le=60)
    additional_features: dict = Field(default_factory=dict)

@app.post("/predict_theft")
def predict_theft(data: ConsumerBehavior):
    if model is None or model_features is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline.py first!")

    try:
        # 2. Map the incoming data to the dictionary
        input_dict = {
            "total_daily_kwh": data.total_daily_kwh,
            "peak_to_offpeak_ratio": data.peak_to_offpeak_ratio,
            "daily_variance": data.daily_variance,
            "temperatureMax": data.temperature_celsius # Mapping temp to the model's name
        }
        
        # Add the extra features from the dashboard's "additional_features"
        input_dict.update(data.additional_features)

        # 3. Create DataFrame and align with model columns
        input_df = pd.DataFrame([input_dict])
        
        # Ensure all columns exist, fill missing with 0
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0.0

        # Order columns exactly as the model expects
        input_df = input_df[model_features]

        # 4. Predict with safety check for single-class edge case
        confidence_scores = model.predict_proba(input_df)[0]
        if len(confidence_scores) > 1:
            thief_probability = confidence_scores[1]
        else:
            thief_probability = confidence_scores[0]

        # Tiered classification based on probability
        probability_percent = thief_probability * 100

        if probability_percent >= 60:
            prediction = "Confirmed Thief"
            risk_level = "HIGH"
            message = "Immediate investigation required. Strong theft indicators detected."
        elif probability_percent >= 40:
            prediction = "Suspicious - Monitor"
            risk_level = "MEDIUM"
            message = "Unusual patterns detected. Recommend closer monitoring."
        else:
            prediction = "Normal User"
            risk_level = "LOW"
            message = "Normal consumption pattern. No anomalies detected."

        return {
            "status": "success",
            "prediction": prediction,
            "risk_level": risk_level,
            "thief_probability": f"{probability_percent:.2f}%",
            "message": message
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Electricity Theft Detection API is active and ready."}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_count": len(model_features) if model_features else 0
    }