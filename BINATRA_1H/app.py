
# =============================================================================
# BINATRA v1H – FLASK API (Realtime Prediction)
# =============================================================================

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import tensorflow as tf
from datetime import datetime
import os

# =============================================================================
# 1. LOAD MODEL + SCALER + CONFIG
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "lstm_banjir_1jam.keras")
scaler_path = os.path.join(BASE_DIR, "scaler_1jam.pkl")
config_path = os.path.join(BASE_DIR, "deploy_config.json")

print("📌 Loading model & scaler from:", BASE_DIR)
model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

with open(config_path, "r") as f:
    config = json.load(f)

SEQ_LEN     = config["SEQ_LEN"]
FEATURES    = config["FEATURES"]
CALIBRATION = config["CALIBRATION"]

print("✅ Model & config loaded!")


# =============================================================================
# 2. PREPROCESSING FUNCTION
# =============================================================================

def preprocess_realtime(df_raw, calibration):
    df = df_raw.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")

    # Clean signals
    df["waterlevel"] = (
        pd.to_numeric(df["waterlevel"], errors="coerce")
        .interpolate()
        .clip(0, calibration - 1)
    )
    df["rainfall"] = (
        pd.to_numeric(df["rainfall"], errors="coerce")
        .clip(lower=0)
        .fillna(0)
    )
    df["voltage"] = (
        pd.to_numeric(df["voltage"], errors="coerce")
        .fillna(method="ffill")
        .fillna(12.5)
    )

    # Target variable
    df["depth_actual"] = calibration - df["waterlevel"]

    # Feature engineering
    df["hour"] = df.index.hour
    df["rain_cum_1h"] = df["rainfall"].rolling(4, min_periods=1).sum()
    df["rain_cum_3h"] = df["rainfall"].rolling(12, min_periods=1).sum()

    return df


def prepare_realtime_input(df_processed, scaler, seq_len=96):
    last = df_processed[FEATURES].tail(seq_len).values

    # Jika kurang dari 96 data → padding
    if last.shape[0] < seq_len:
        pad = np.repeat(last[0:1], seq_len - last.shape[0], axis=0)
        last = np.vstack([pad, last])

    scaled = scaler.transform(last)
    return scaled.reshape(1, seq_len, len(FEATURES))


def inverse_scale_depth(pred_scaled, scaler):
    dummy = np.zeros((len(pred_scaled), len(FEATURES)))
    dummy[:, 0] = pred_scaled
    inv = scaler.inverse_transform(dummy)[:, 0]
    return inv.tolist()


# =============================================================================
# 3. FLASK APP
# =============================================================================

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "BINATRA_1H API Running",
        "message": "Use /predict (POST) to get predictions"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expecting a list of dicts
        df = pd.DataFrame(data)

        # Preprocess
        df_processed = preprocess_realtime(df, CALIBRATION)

        # Prepare input for model
        X_input = prepare_realtime_input(df_processed, scaler, SEQ_LEN)

        # Predict
        pred_scaled = model.predict(X_input)[0][:, 0]
        pred_cm = inverse_scale_depth(pred_scaled, scaler)

        return jsonify({
            "status": "success",
            "prediction_cm": pred_cm,
            "horizon_minutes": 60,
            "step_minutes": 15
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK", "model": "BINATRA v1H"})


# =============================================================================
# 4. RUN SERVER
# =============================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
