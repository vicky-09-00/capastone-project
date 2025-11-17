import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

def generate_synthetic_data():
    """Generate synthetic heart disease dataset for training"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data based on typical heart disease patterns
    data = {
        'age': np.random.randint(29, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),  # 0 = female, 1 = male
        'cp': np.random.randint(0, 4, n_samples),  # Chest pain type
        'trestbps': np.random.randint(94, 200, n_samples),  # Resting blood pressure
        'chol': np.random.randint(126, 564, n_samples),  # Serum cholesterol
        'fbs': np.random.randint(0, 2, n_samples),  # Fasting blood sugar
        'restecg': np.random.randint(0, 3, n_samples),  # Resting ECG
        'thalach': np.random.randint(71, 202, n_samples),  # Max heart rate
        'exang': np.random.randint(0, 2, n_samples),  # Exercise induced angina
        'oldpeak': np.random.uniform(0, 6.2, n_samples),  # ST depression
        'slope': np.random.randint(0, 3, n_samples),  # Slope of peak exercise
        'ca': np.random.randint(0, 4, n_samples),  # Number of major vessels
        'thal': np.random.randint(0, 3, n_samples),  # Thalassemia
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable with some logic
    # Higher risk factors increase probability of heart disease
    risk_score = (
        (df['age'] > 50).astype(int) * 0.15 +  # Lower age threshold
        (df['trestbps'] > 130).astype(int) * 0.15 +  # Lower BP threshold
        (df['chol'] > 200).astype(int) * 0.15 +  # Lower cholesterol threshold
        (df['thalach'] < 150).astype(int) * 0.15 +  # More lenient heart rate
        (df['oldpeak'] > 1).astype(int) * 0.15 +  # Lower ST depression threshold
        (df['exang'] == 1).astype(int) * 0.15 +  # Exercise angina
        (df['cp'] > 0).astype(int) * 0.1 +  # Any chest pain
        (df['ca'] > 0).astype(int) * 0.1 +  # Blocked vessels
        (df['thal'] > 0).astype(int) * 0.1  # Thalassemia issues
    )
    
    # Add some randomness
    risk_score += np.random.uniform(-0.15, 0.15, n_samples)
    # Lower threshold to 0.35 to detect more cases
    df['target'] = (risk_score > 0.35).astype(int)
    
    # Ensure at least 40% have heart disease for balanced dataset
    positive_count = df['target'].sum()
    if positive_count < n_samples * 0.4:
        # Add more positive cases
        negative_indices = df[df['target'] == 0].index
        needed = int(n_samples * 0.4) - positive_count
        if needed > 0 and len(negative_indices) > 0:
            flip_indices = np.random.choice(negative_indices, min(needed, len(negative_indices)), replace=False)
            df.loc[flip_indices, 'target'] = 1
    
    # Classify disease types based on symptoms
    # 0 = No Disease, 1 = Coronary Artery Disease, 2 = Arrhythmia, 
    # 3 = Heart Failure, 4 = Valvular Heart Disease, 5 = Cardiomyopathy
    disease_type = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        if df.loc[i, 'target'] == 1:  # If heart disease is present
            # Coronary Artery Disease (high cholesterol, chest pain, blocked vessels)
            if df.loc[i, 'chol'] > 240 and df.loc[i, 'cp'] > 1 and df.loc[i, 'ca'] > 0:
                disease_type[i] = 1
            # Arrhythmia (irregular heart rate, ECG abnormalities)
            elif df.loc[i, 'restecg'] > 0 and (df.loc[i, 'thalach'] < 100 or df.loc[i, 'thalach'] > 180):
                disease_type[i] = 2
            # Heart Failure (low heart rate, high BP, exercise intolerance)
            elif df.loc[i, 'thalach'] < 120 and df.loc[i, 'trestbps'] > 140 and df.loc[i, 'exang'] == 1:
                disease_type[i] = 3
            # Valvular Heart Disease (thalassemia defects, slope issues)
            elif df.loc[i, 'thal'] > 0 and df.loc[i, 'slope'] == 2:
                disease_type[i] = 4
            # Cardiomyopathy (ST depression, multiple vessels affected)
            elif df.loc[i, 'oldpeak'] > 2 and df.loc[i, 'ca'] >= 2:
                disease_type[i] = 5
            else:
                # Default to Coronary Artery Disease if unclear
                disease_type[i] = 1
    
    df['disease_type'] = disease_type
    
    return df

def train_model():
    """Train and save the heart disease prediction model"""
    # Generate or load dataset
    df = generate_synthetic_data()
    
    # Separate features and target
    X = df.drop(['target', 'disease_type'], axis=1)
    y = df['target']
    y_type = df['disease_type']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model for binary classification
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Train model for disease type classification (only on positive cases)
    positive_mask = y_train == 1
    X_type_train = X_train_scaled[positive_mask]
    y_type_train = y_type.iloc[X_train.index[positive_mask]]
    
    if len(X_type_train) > 0:
        type_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        type_model.fit(X_type_train, y_type_train)
    else:
        type_model = None
    
    # Calculate accuracy
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"Training Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    
    # Save models and scaler
    with open('heart_disease_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    if type_model is not None:
        with open('disease_type_model.pkl', 'wb') as f:
            pickle.dump(type_model, f)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return model, scaler, type_model

def predict_heart_disease(features, model=None, scaler=None, type_model=None):
    """Predict heart disease from features"""
    if model is None:
        with open('heart_disease_model.pkl', 'rb') as f:
            model = pickle.load(f)
    
    if scaler is None:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # Predict disease type if heart disease is detected
    disease_type = 0
    type_probability = None
    if prediction == 1 and type_model is not None:
        try:
            if os.path.exists('disease_type_model.pkl'):
                if type_model is None:
                    with open('disease_type_model.pkl', 'rb') as f:
                        type_model = pickle.load(f)
                disease_type = type_model.predict(features_scaled)[0]
                type_probability = type_model.predict_proba(features_scaled)[0]
        except:
            # Fallback to rule-based classification
            disease_type = classify_disease_type(features)
    
    return prediction, probability, disease_type, type_probability

def classify_disease_type(features):
    """Rule-based classification of heart disease type"""
    age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = features
    
    # Coronary Artery Disease (high cholesterol, chest pain, blocked vessels)
    if chol > 240 and cp > 1 and ca > 0:
        return 1
    # Arrhythmia (irregular heart rate, ECG abnormalities)
    elif restecg > 0 and (thalach < 100 or thalach > 180):
        return 2
    # Heart Failure (low heart rate, high BP, exercise intolerance)
    elif thalach < 120 and trestbps > 140 and exang == 1:
        return 3
    # Valvular Heart Disease (thalassemia defects, slope issues)
    elif thal > 0 and slope == 2:
        return 4
    # Cardiomyopathy (ST depression, multiple vessels affected)
    elif oldpeak > 2 and ca >= 2:
        return 5
    else:
        # Default to Coronary Artery Disease
        return 1

if __name__ == '__main__':
    train_model()

