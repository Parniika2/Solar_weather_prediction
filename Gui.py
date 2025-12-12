import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd

# Load Model
try:
    model = joblib.load('solar_rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    messagebox.showerror("Error", "Model files not found! Run the notebook first.")
    exit()

def predict_energy():
    try:
        # Get values from input boxes
        ghi = float(entry_ghi.get())
        temp = float(entry_temp.get())
        hum = float(entry_hum.get())
        wind = float(entry_wind.get())
        press = float(entry_press.get())
        cloud = float(entry_cloud.get())
        
        # Prepare Data
        feature_cols = ['GHI', 'temp', 'humidity', 'wind_speed', 'pressure', 'clouds_all']
        data = pd.DataFrame([[ghi, temp, hum, wind, press, cloud]], columns=feature_cols)
        
        # Scale
        data_scaled = scaler.transform(data)
        
        # Predict
        pred = model.predict(data_scaled)[0]
        
        # Show Result
        lbl_result.config(text=f"Predicted Energy: {pred:.2f} Wh", fg="green")
        
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

# --- GUI Setup ---
root = tk.Tk()
root.title("Solar Energy Predictor")
root.geometry("400x500")

tk.Label(root, text="Solar Power Prediction", font=("Arial", 16, "bold")).pack(pady=10)

# Input Fields
frame = tk.Frame(root)
frame.pack(pady=10)

labels = ["GHI (W/m²)", "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Pressure (hPa)", "Cloud Cover (%)"]
entries = []

# GHI
tk.Label(frame, text=labels[0]).grid(row=0, column=0, padx=5, pady=5)
entry_ghi = tk.Entry(frame)
entry_ghi.grid(row=0, column=1)

# Temp
tk.Label(frame, text=labels[1]).grid(row=1, column=0, padx=5, pady=5)
entry_temp = tk.Entry(frame)
entry_temp.grid(row=1, column=1)

# Humidity
tk.Label(frame, text=labels[2]).grid(row=2, column=0, padx=5, pady=5)
entry_hum = tk.Entry(frame)
entry_hum.grid(row=2, column=1)

# Wind
tk.Label(frame, text=labels[3]).grid(row=3, column=0, padx=5, pady=5)
entry_wind = tk.Entry(frame)
entry_wind.grid(row=3, column=1)

# Pressure
tk.Label(frame, text=labels[4]).grid(row=4, column=0, padx=5, pady=5)
entry_press = tk.Entry(frame)
entry_press.grid(row=4, column=1)

# Cloud
tk.Label(frame, text=labels[5]).grid(row=5, column=0, padx=5, pady=5)
entry_cloud = tk.Entry(frame)
entry_cloud.grid(row=5, column=1)

# Button
btn_predict = tk.Button(root, text="Predict", command=predict_energy, bg="blue", fg="white", font=("Arial", 12))
btn_predict.pack(pady=20)

# Result Label
lbl_result = tk.Label(root, text="Predicted Energy: ---", font=("Arial", 14))
lbl_result.pack()

root.mainloop()