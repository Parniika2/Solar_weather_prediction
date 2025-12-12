"""
Train a RandomForest model on `solar_weather.csv` and save model + scaler as .pkl files
Run: python train_save_model.py
"""
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

CSV = 'solar_weather.csv'
MODEL_OUT = 'solar_rf_model.pkl'
SCALER_OUT = 'scaler.pkl'

FEATURE_COLS = ['GHI', 'temp', 'humidity', 'wind_speed', 'pressure', 'clouds_all']
TARGET_COL = 'Energy delta[Wh]'


def main():
    try:
        df = pd.read_csv(CSV)
    except FileNotFoundError:
        print(f"Error: CSV file '{CSV}' not found in the current directory.")
        sys.exit(1)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        print("Error: The following required columns are missing from the CSV:", missing)
        sys.exit(1)

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # quick evaluation
    preds = model.predict(X_test_scaled)
    rmse = (mean_squared_error(y_test, preds)) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    print(f"Trained RandomForest - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # save artifacts
    joblib.dump(model, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    print(f"Saved model to {MODEL_OUT} and scaler to {SCALER_OUT}")


if __name__ == '__main__':
    main()
