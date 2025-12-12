import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load Model and Scaler
# Use @st.cache_resource to load them only once for speed
@st.cache_resource
def load_models():
    try:
        model = joblib.load('solar_rf_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found! Please run the Jupyter Notebook first to generate .pkl files.")
        return None, None

model, scaler = load_models()

# 2. App Title and Description
st.title("☀️ Solar Power Generation Predictor")
st.markdown("""
This app predicts the solar energy generation (*Energy delta[Wh]*) based on meteorological conditions.
Please adjust the parameters below.
""")

if model is not None:
    # 3. Sidebar Inputs
    st.sidebar.header("Meteorological Inputs")

    # Define inputs matching the training feature order:
    # ['GHI', 'temp', 'humidity', 'wind_speed', 'pressure', 'clouds_all']
    
    ghi = st.sidebar.slider("Global Horizontal Irradiance (GHI)", 0.0, 1200.0, 500.0)
    temp = st.sidebar.slider("Temperature (°C)", -20.0, 50.0, 25.0)
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0)
    pressure = st.sidebar.slider("Pressure (hPa)", 900.0, 1100.0, 1013.0)
    clouds_all = st.sidebar.slider("Cloud Cover (%)", 0.0, 100.0, 20.0)

    # 4. Predict Button
    if st.button("Predict Solar Power"):
        # Create a dataframe for the input
        input_data = pd.DataFrame([[ghi, temp, humidity, wind_speed, pressure, clouds_all]],
                                  columns=['GHI', 'temp', 'humidity', 'wind_speed', 'pressure', 'clouds_all'])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Display Result
        st.success(f"Predicted Energy Generation: *{prediction:.2f} Wh*")
        
        # Visualizing where the prediction falls
        st.metric(label="Energy Output", value=f"{prediction:.2f} Wh")
        st.progress(min(int(prediction)/2000, 1.0)) # Assuming max around 2000 Wh for bar scaling