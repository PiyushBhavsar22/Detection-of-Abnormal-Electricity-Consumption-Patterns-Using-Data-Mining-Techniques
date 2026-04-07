import streamlit as st
import requests

# Set up the page
st.set_page_config(page_title="Electricity Theft Detector", page_icon="⚡", layout="centered")

st.title("⚡ AI Electricity Theft Detection")
st.markdown("Enter consumer metrics to score them in real-time. (Feature Mapping Active)")

st.divider()

# Create sliders and inputs using the EXACT model feature names
col1, col2 = st.columns(2)

with col1:
    total_daily_kwh = st.number_input("Total Daily Usage (total_daily_kwh)", min_value=0.0, max_value=500.0, value=15.0)
    daily_variance = st.number_input("Hourly Variance (daily_variance)", min_value=0.0, max_value=100.0, value=2.5)
# Change the max_value from 200.0 to 1000.0
    peak_sum = st.number_input("Peak Usage Sum (peak_sum)", min_value=0.0, max_value=1000.0, value=5.0)
    off_peak_sum = st.number_input("Off-Peak Usage Sum (off_peak_sum)", min_value=0.0, max_value=1000.0, value=10.0)

with col2:
    peak_to_offpeak_ratio = st.slider("Peak/Off-Peak Ratio", min_value=0.0, max_value=20.0, value=1.2)
    temperatureMax = st.slider("Max Daily Temp (temperatureMax)", min_value=-10.0, max_value=50.0, value=18.0)
    temp_hr_std = st.slider("Temp Hourly Std Dev (temp_hr_std)", min_value=0.0, max_value=15.0, value=2.0)
    is_holiday = st.checkbox("Is it a Holiday? (is_holiday)")

st.divider()

if st.button("🔍 Run Fraud Analysis", type="primary", use_container_width=True):
    
    # We map the UI variables directly to the keys the model_features.pkl expects
    payload = {
        "total_daily_kwh": total_daily_kwh,
        "peak_to_offpeak_ratio": peak_to_offpeak_ratio,
        "daily_variance": daily_variance,
        "temperature_celsius": temperatureMax, # The validator checks this, API will rename to temperatureMax
        "additional_features": {
            "peak_sum": peak_sum,
            "off_peak_sum": off_peak_sum,
            "temperatureMax": temperatureMax,
            "temp_hr_std": temp_hr_std,
            "is_holiday": int(is_holiday)
        }
    }
    
    try:
        with st.spinner('AI is analyzing behavioral patterns...'):
            response = requests.post("http://127.0.0.1:8000/predict_theft", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                if result["prediction"] == "Confirmed Thief":
                    st.error(f"🚨 **ALERT: {result['prediction']}**")
                else:
                    st.success(f"✅ **{result['prediction']}**")
                    
                st.metric(label="AI Confidence Score", value=result["thief_probability"])
                st.info(result["message"])
            else:
                st.error(f"API Error: {response.text}")
                
    except requests.exceptions.ConnectionError:
        st.error("🚨 API Offline. Run 'uvicorn api:app' in your terminal!")