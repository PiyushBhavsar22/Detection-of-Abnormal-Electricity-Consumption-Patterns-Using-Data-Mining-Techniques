from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Electricity Theft Detection API")

# Load model and features
model = joblib.load("theft_detection_model.pkl")
model_features = joblib.load("model_features.pkl")

# 1. Update the Validator to match the actual feature names
class ConsumerBehavior(BaseModel):
    total_daily_kwh: float = Field(..., ge=0)
    peak_to_offpeak_ratio: float = Field(..., ge=0)
    daily_variance: float = Field(..., ge=0) # Renamed from variance_kwh
    temperature_celsius: float = Field(..., ge=-50, le=60)
    additional_features: dict = Field(default_factory=dict)

@app.post("/predict_theft")
def predict_theft(data: ConsumerBehavior):
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

        # 4. Predict
        confidence_scores = model.predict_proba(input_df)[0]
        thief_probability = confidence_scores[1]
        is_thief = bool(thief_probability >= 0.75)

        return {
            "status": "success",
            "prediction": "Confirmed Thief" if is_thief else "Normal User",
            "thief_probability": f"{thief_probability * 100:.2f}%",
            "message": "Flagged for manual inspection." if is_thief else "Normal consumption pattern."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "API is online"}