from fastapi.testclient import TestClient
from api import app

# Initialize the simulated client
client = TestClient(app)

def test_api_health():
    """Test 1: Does the API wake up correctly?"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Electricity Theft Detection API is active and ready."}
    print("✅ Health Check Passed!")

def test_normal_behavior():
    """Test 2: Can the API process a normal household without crashing?"""
    payload = {
        "total_daily_kwh": 12.5,
        "peak_to_offpeak_ratio": 1.1,
        "daily_variance": 2.4,
        "temperature_celsius": 18.0,
        "additional_features": {} # Missing demographics default to 0 gracefully
    }
    response = client.post("/predict_theft", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    print(f"✅ Normal Behavior Test Passed! Prediction: {response.json()['prediction']}")

def test_thief_behavior():
    """Test 3: Can the API process extreme behavior?"""
    payload = {
        "total_daily_kwh": 150.5, # Highly suspicious usage
        "peak_to_offpeak_ratio": 8.5, # Massive spikes
        "daily_variance": 45.0,
        "temperature_celsius": 22.0,
        "additional_features": {}
    }
    response = client.post("/predict_theft", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    print(f"✅ Extreme Behavior Test Passed! Prediction: {response.json()['prediction']}")

def test_validator_negative_usage():
    """Test 4: THE TRAP. Will the API reject physically impossible negative electricity?"""
    payload = {
        "total_daily_kwh": -50.0, # IMPOSSIBLE!
        "peak_to_offpeak_ratio": 1.1,
        "daily_variance": 2.4,
        "temperature_celsius": 18.0
    }
    response = client.post("/predict_theft", json=payload)
    # 422 is the standard HTTP code for "Unprocessable Entity" (Validation Failed)
    assert response.status_code == 422
    print("✅ Validator Test Passed! The API successfully blocked negative energy usage.")

def test_validator_extreme_weather():
    """Test 5: THE TRAP. Will the API reject impossible weather inputs?"""
    payload = {
        "total_daily_kwh": 15.0,
        "peak_to_offpeak_ratio": 1.1,
        "daily_variance": 2.4,
        "temperature_celsius": 200.0 # IMPOSSIBLE ON EARTH!
    }
    response = client.post("/predict_theft", json=payload)
    assert response.status_code == 422
    print("✅ Weather Validator Passed! The API blocked unrealistic temperatures.")