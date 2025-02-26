from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

# Load trained model
model = joblib.load("random_forest_model.joblib")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("front_page.html")  # Load the form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        temperature = int(request.form['temperature'])
        humidity = int(request.form['humidity'])
        moisture = int(request.form['moisture'])
        soil_type = request.form['soil']
        crop_type = request.form['crop']
        nitrogen = int(request.form['nitrogen'])
        potassium = int(request.form['potassium'])
        phosphorous = int(request.form['phosphorous'])

        print(f"Received Input - Temp: {temperature}, Humidity: {humidity}, Moisture: {moisture}, Soil: {soil_type}, Crop: {crop_type}, N: {nitrogen}, K: {potassium}, P: {phosphorous}")

        # Convert categorical values (Soil, Crop) to numeric if necessary
        soil_mapping = {'sandy': 0, 'clay': 1, 'loamy': 2}
        crop_mapping = {'wheat': 0, 'rice': 1, 'maize': 2, 'sugarcane': 3}

        soil_value = soil_mapping.get(soil_type, 0)
        crop_value = crop_mapping.get(crop_type, 0)

        # **Ensure feature names match exactly as in training**
        feature_names = ["Temperature", "Humidity", "Moisture", "Soil", "Crop", "nitrogen", "Potassium", "Phosphorous"]
        features = pd.DataFrame([[temperature, humidity, moisture, soil_value, crop_value, nitrogen, potassium, phosphorous]], columns=feature_names)

        print("Prepared Features for Prediction:", features)

        # Make prediction
        prediction = model.predict(features)[0]

        print("🔹 Model Prediction:", prediction)  # Debugging print

        return render_template("front_page.html", prediction_text=f"Recommended Fertilizer: {prediction}")

    except Exception as e:
        print("❌ Error:", str(e))
        return render_template("front_page.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not set
    app.run(host='0.0.0.0', port=port, debug=False)
