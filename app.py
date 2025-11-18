from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
from model import train_model, predict_heart_disease, classify_disease_type

app = Flask(__name__)

# Train model on startup if model doesn't exist
MODEL_PATH = 'heart_disease_model.pkl'
SCALER_PATH = 'scaler.pkl'
TYPE_MODEL_PATH = 'disease_type_model.pkl'
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("Training model...")
    train_model()
    print("Model trained and saved!")

# Load the trained model and scaler
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# Load disease type model if it exists
type_model = None
if os.path.exists(TYPE_MODEL_PATH):
    with open(TYPE_MODEL_PATH, 'rb') as f:
        type_model = pickle.load(f)

# Disease type mapping
DISEASE_TYPES = {
    0: "No Heart Disease",
    1: "Coronary Artery Disease (CAD)",
    2: "Arrhythmia",
    3: "Heart Failure",
    4: "Valvular Heart Disease",
    5: "Cardiomyopathy"
}

DISEASE_DESCRIPTIONS = {
    1: "Blockage or narrowing of coronary arteries, often caused by high cholesterol and plaque buildup.",
    2: "Irregular heart rhythm, detected through ECG abnormalities and abnormal heart rate patterns.",
    3: "Heart's inability to pump blood effectively, often indicated by low heart rate and exercise intolerance.",
    4: "Problems with heart valves, often related to structural defects and blood flow issues.",
    5: "Disease of the heart muscle, often indicated by ST depression and multiple vessel involvement."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features in the correct order
        features = [
            float(data.get('age', 0)),
            float(data.get('sex', 0)),
            float(data.get('cp', 0)),
            float(data.get('trestbps', 0)),
            float(data.get('chol', 0)),
            float(data.get('fbs', 0)),
            float(data.get('restecg', 0)),
            float(data.get('thalach', 0)),
            float(data.get('exang', 0)),
            float(data.get('oldpeak', 0)),
            float(data.get('slope', 0)),
            float(data.get('ca', 0)),
            float(data.get('thal', 0))
        ]
        
        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features using the same scaler used during training
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Get probability of heart disease
        heart_disease_prob = probability[1] * 100 if len(probability) > 1 else probability[0] * 100
        
        # Rule-based check: If ML says no but symptoms suggest yes, override
        # Check for multiple risk factors
        risk_factors = 0
        if float(data.get('age', 0)) > 50:
            risk_factors += 1
        if float(data.get('trestbps', 0)) > 130:
            risk_factors += 1
        if float(data.get('chol', 0)) > 200:
            risk_factors += 1
        if float(data.get('thalach', 0)) < 150:
            risk_factors += 1
        if float(data.get('oldpeak', 0)) > 1:
            risk_factors += 1
        if float(data.get('exang', 0)) == 1:
            risk_factors += 1
        if float(data.get('cp', 0)) > 0:
            risk_factors += 1
        if float(data.get('ca', 0)) > 0:
            risk_factors += 1
        
        # If 3+ risk factors, likely heart disease
        if prediction == 0 and risk_factors >= 3:
            prediction = 1
            heart_disease_prob = min(75 + (risk_factors - 3) * 5, 95)  # Adjust probability
        
        # If probability is borderline (30-50%), check symptoms
        if prediction == 0 and 30 <= heart_disease_prob <= 50 and risk_factors >= 2:
            prediction = 1
            heart_disease_prob = 55 + (risk_factors - 2) * 5
        
        # Determine disease type if heart disease is detected
        disease_type = 0
        disease_type_name = "No Heart Disease"
        disease_description = ""
        
        if prediction == 1:
            # Try to predict disease type using ML model
            if type_model is not None:
                try:
                    disease_type = type_model.predict(features_scaled)[0]
                    disease_type_name = DISEASE_TYPES.get(disease_type, "Heart Disease")
                    disease_description = DISEASE_DESCRIPTIONS.get(disease_type, "")
                except:
                    # Fallback to rule-based classification
                    disease_type = classify_disease_type(features)
                    disease_type_name = DISEASE_TYPES.get(disease_type, "Heart Disease")
                    disease_description = DISEASE_DESCRIPTIONS.get(disease_type, "")
            else:
                # Use rule-based classification
                disease_type = classify_disease_type(features)
                disease_type_name = DISEASE_TYPES.get(disease_type, "Heart Disease")
                disease_description = DISEASE_DESCRIPTIONS.get(disease_type, "")
        
        # Ensure probability is reasonable
        if prediction == 1:
            heart_disease_prob = max(heart_disease_prob, 50)  # Minimum 50% if detected
        else:
            heart_disease_prob = min(heart_disease_prob, 30)  # Maximum 30% if not detected
        
        result = {
            'prediction': int(prediction),
            'probability': round(heart_disease_prob, 2),
            'status': 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease',
            'message': f'Risk Level: {round(heart_disease_prob, 2)}%',
            'disease_type': disease_type,
            'disease_type_name': disease_type_name,
            'disease_description': disease_description
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

