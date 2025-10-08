from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS  # Fix CORS issue

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# ✅ Load Models & Scalers
crop_model = joblib.load("models/crop_model.pkl")
crop_scaler = joblib.load("models/crop_scaler.pkl")

fertilizer_model = joblib.load("models/fertilizer_model.pkl")  # FIXED: Ensure it's a model
fertilizer_scaler = joblib.load("models/fertilizer_scaler.pkl")
fertilizer_encoder = joblib.load("models/fertilizer_encoder.pkl")

# ✅ Ensure scaler is a StandardScaler
if not isinstance(crop_scaler, StandardScaler):
    raise TypeError(f"Expected StandardScaler but got {type(crop_scaler)}")
if not isinstance(fertilizer_scaler, StandardScaler):
    raise TypeError(f"Expected StandardScaler but got {type(fertilizer_scaler)}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        data = request.get_json()

        # ✅ Convert input to array
        features = np.array([[data["N"], data["P"], data["K"], data["temp"], data["hum"], data["ph"], data["rain"]]])

        # ✅ Ensure correct input format
        features_scaled = crop_scaler.transform(features)

        # ✅ Predict crop
        prediction = crop_model.predict(features_scaled)[0]

        return jsonify({"Recommended Crop": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/recommend_fertilizer", methods=["POST"])
def recommend_fertilizer():
    try:
        data = request.get_json()

        # ✅ Fix: Correct key name from "Temparature" to "Temperature"
        encoded_features = fertilizer_encoder.transform([[data["Soil Type"], data["Crop Type"]]])

        # ✅ Convert to numpy array
        features = np.hstack((encoded_features, np.array([[data["Temparature"], data["Humidity"], 
                                                           data["Moisture"], data["Nitrogen"], 
                                                           data["Potassium"], data["Phosphorous"]]])))

        # ✅ Ensure proper shape before scaling
        features = features.reshape(1, -1)

        # ✅ Scale features correctly
        features_scaled = fertilizer_scaler.transform(features)

        # ✅ Ensure model is valid
        if not hasattr(fertilizer_model, "predict"):
            raise ValueError("Fertilizer model is not a valid ML model. Re-check the saved model.")

        # ✅ Predict Fertilizer
        prediction = fertilizer_model.predict(features_scaled)[0]

        return jsonify({"Recommended Fertilizer": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
