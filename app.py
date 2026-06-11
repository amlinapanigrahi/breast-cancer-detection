import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='templates', static_folder='static')


MODEL_PATH = 'best_model.pkl'
SCALER_PATH = 'scaler.pkl'
DATA_PATH = 'data.csv'


model = None
scaler = None


FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

def load_resources():
    global model, scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return False
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return True

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, scaler
    if model is None or scaler is None:
        if not load_resources():
            return jsonify({'error': 'Model or scaler file not found. Please train and save the model in the notebook first.'}), 500
            
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        input_features = []
        for feat in FEATURE_NAMES:
            if feat not in data:
                return jsonify({'error': f'Missing feature: {feat}'}), 400
            input_features.append(float(data[feat]))
            
        features_array = np.array(input_features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)
        
        
        prediction = int(model.predict(scaled_features)[0])
        probabilities = model.predict_proba(scaled_features)[0]
        
        class_label = 'Malignant' if prediction == 1 else 'Benign'
        confidence = float(probabilities[prediction]) * 100
        
        return jsonify({
            'prediction': class_label,
            'confidence': confidence,
            'probabilities': {
                'Benign': float(probabilities[0]) * 100,
                'Malignant': float(probabilities[1]) * 100
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/random-sample', methods=['GET'])
def get_random_sample():
    try:
        if not os.path.exists(DATA_PATH):
            return jsonify({'error': 'dataset file data.csv not found'}), 404
            
        
        df = pd.read_csv(DATA_PATH)
        

        if 'Unnamed: 32' in df.columns:
            df = df.drop('Unnamed: 32', axis=1)
            
       
        random_row = df.sample(n=1).iloc[0]
        
       
        sample_data = random_row.to_dict()
        
        actual_diagnosis = 'Malignant' if sample_data['diagnosis'] in ['M', 1] else 'Benign'
        
        return jsonify({
            'actual_diagnosis': actual_diagnosis,
            'features': {k: v for k, v in sample_data.items() if k not in ['id', 'diagnosis']}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Loading model and scaler...")
    load_resources()
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
