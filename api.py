from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Model and Scaler
try:
    model = joblib.load('solar_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and Scaler loaded successfully.")
except:
    print("Error: Model files not found. Run the notebook first.")

@app.route('/')
def home():
    return "Solar Power Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features
        # Expecting JSON keys matching training columns
        features = [
            data['GHI'],
            data['temp'],
            data['humidity'],
            data['wind_speed'],
            data['pressure'],
            data['clouds_all']
        ]
        
        # Convert to DataFrame
        feature_cols = ['GHI', 'temp', 'humidity', 'wind_speed', 'pressure', 'clouds_all']
        df_input = pd.DataFrame([features], columns=feature_cols)
        
        # Scale
        input_scaled = scaler.transform(df_input)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'status': 'success',
            'predicted_energy_wh': float(prediction)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

# --- Example Curl Request ---
# curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"GHI\": 600, \"temp\": 30, \"humidity\": 45, \"wind_speed\": 3.5, \"pressure\": 1012, \"clouds_all\": 10}"