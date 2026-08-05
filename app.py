from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
import os
import pandas as pd
import joblib
from weather_api import get_weather
from disease_info import disease_info

app = Flask(__name__)

# ===========================
# LOAD TRAINED MODELS
# ===========================
import joblib, os
MODEL_DIR = "models"
model_path = os.path.join(MODEL_DIR, "rf_model.joblib")
print("Model exists:", os.path.exists(model_path))



# Load ML models
rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib"))


fertilizer_model = pickle.load(open(os.path.join("models", "fertilizer_model.pkl"), "rb"))
print("✅ Fertilizer model loaded successfully!")

disease_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "disease_model.h5"))

# Load encoders and label mappings
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))

with open(os.path.join(MODEL_DIR, "class_indices.json")) as f:
    class_indices = json.load(f)
classes = list(class_indices.keys())

# ===========================
# ROUTES
# ===========================
@app.route('/')
def home():
    return render_template('index.html')

# ---------- Crop Recommendation ----------
@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        ph = float(request.form['ph'])
        city = request.form['city']
        season = request.form['season']

        # --- Get weather data ---
        temperature, humidity, rainfall = get_weather(city)

        if temperature is None:
            return jsonify({'error': 'Weather data not available. Check city name.'})

        # --- Fallback rainfall by season ---
        if rainfall == 0 or rainfall is None:
            avg_rainfall_by_season = {
                "Kharif": 250,
                "Rabi": 50,
                "Summer": 120,
                "Winter": 40,
                "Monsoon": 300
            }
            rainfall = avg_rainfall_by_season.get(season.capitalize(), 100)
            print(f"Rainfall missing, using fallback: {rainfall} mm")

        # --- Prepare input ---
        features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

        # --- Predict crop ---
        rf_pred = rf_model.predict(features)[0]
        crop_name = label_encoder.inverse_transform([int(rf_pred)])[0]

        return render_template('result.html',
                               prediction_text=f"🌾 Recommended Crop: {crop_name}",
                               temperature=temperature, humidity=humidity,
                               rainfall=rainfall, season=season)

    except Exception as e:
        print("Error during crop prediction:", e)
        return jsonify({'error': str(e)})


# ---------- Fertilizer Recommendation ----------
@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html')

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        temp = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        moisture = float(request.form["moisture"])
        soil = request.form["soil"]
        crop = request.form["crop"]
        nitrogen = float(request.form["nitrogen"])
        potassium = float(request.form["potassium"])
        phosphorus = float(request.form["phosphorus"])

        # Create a dataframe for the input
        input_data = pd.DataFrame({
            "Temperature": [temp],
            "Humidity": [humidity],
            "Moisture": [moisture],
            "Soil Type": [soil],
            "Crop Type": [crop],
            "Nitrogen": [nitrogen],
            "Potassium": [potassium],
            "Phosphorous": [phosphorus]
        })

        # Match one-hot encoding
        input_data = pd.get_dummies(input_data)
        model_columns = fertilizer_model.feature_names_in_
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        prediction = fertilizer_model.predict(input_data)[0]
        print("Prediction:", prediction)

        # ✅ Add Fertilizer Tips Dictionary
        fertilizer_tips = {
            "Urea": "Urea is rich in nitrogen and helps in leafy growth. Use it for crops like wheat, maize, and rice.",
            "DAP": "DAP (Diammonium Phosphate) improves phosphorus levels and promotes strong root growth.",
            "14-35-14": "This fertilizer provides balanced nutrients and enhances early plant growth.",
            "28-28": "Good for initial stages of crop development and helps boost nitrogen and phosphorus levels.",
            "20-20": "Balanced NPK fertilizer suitable for general crop nourishment.",
            "10-26-26": "Ideal for flowering and fruiting stages; promotes stronger yield.",
            "17-17-17": "Promotes uniform growth and helps in maintaining soil fertility.",
            "Ammonium Sulphate": "Improves nitrogen and sulphur levels in soil, beneficial for rice and onion.",
            "MOP": "Muriate of Potash strengthens roots and improves water retention in plants."
        }

        # Get explanation text if exists
        explanation = fertilizer_tips.get(prediction, "This fertilizer helps improve soil fertility and crop yield.")

        # Pass both to result.html
        return render_template(
            "result.html",
            prediction_text=f"Recommended Fertilizer: {prediction}",
            explanation_text=explanation
        )
    except Exception as e:
        print("Error during fertilizer prediction:", e)
        return jsonify({'error': str(e)})


# ---------- Disease Prediction ----------
@app.route('/disease')
def disease():
    return render_template('disease.html')

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if "file" not in request.files:
        return "No file uploaded!"
    file = request.files["file"]
    if file.filename == "":
        return "No image selected!"

    # Save uploaded file
    filepath = os.path.join("static/uploads", file.filename)
    os.makedirs("static/uploads", exist_ok=True)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    img_array /= 255.0
    # Predict disease
    preds = disease_model.predict(img_array)
    pred_class = classes[np.argmax(preds)].replace("__", "_").replace("___", "_").strip()
    pred_class = pred_class.lower()


    

    # ✅ Step 2: Get details safely
    details = disease_info.get(pred_class, {
        "Cause": "Unknown",
        "Prevention": "No data available.",
        "Cure": "No information found."
    })
    print(f"Predicted class: {pred_class}")

    # ✅ Step 3: Send to result.html
    return render_template(
        "result.html",
        prediction_text=f"Disease Detected: {pred_class.replace('_', ' ')}",
        cause_text=details["Cause"],
        prevention_text=details["Prevention"],
        cure_text=details["Cure"],
        image_path=file.filename
    )
    
# ===========================
# RUN APP
# ===========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)