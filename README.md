# SolarPowerPrediction

This repository contains a small project for predicting solar energy generation from meteorological features.

Files included:
- `Solar_Power_Prediction.ipynb` - Notebook used for exploration and training.
- `train_save_model.py` - Script to train a RandomForest and save model + scaler (`solar_rf_model.pkl`, `scaler.pkl`).
- `app.py` - Streamlit app UI for interactive prediction.
- `api.py` - Flask API exposing a `/predict` endpoint.
- `Gui.py` - Tkinter desktop GUI for quick predictions.
- `solar_weather.csv` - Dataset used for training (if present).
- `requirements.txt` - Python dependencies.

Quick start

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Train the model (creates `solar_rf_model.pkl` and `scaler.pkl`):

```powershell
python train_save_model.py
```

4. Run the Streamlit app:

```powershell
streamlit run app.py
```

5. Or run the API:

```powershell
python api.py
```

Notes
- Model artifacts (`*.pkl`) are ignored by default and should be generated via the training script.
- If you want to include model files in the repository, consider using Git LFS.
