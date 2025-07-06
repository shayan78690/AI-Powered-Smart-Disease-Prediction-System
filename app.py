from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from model_training import DiseasePredictionModel

app = Flask(__name__)

# Global variables
model = None
descriptions = {}
precautions = {}
available_symptoms = []

def clean_text_column(series):
    """Clean text data: replace underscores, strip spaces, lowercase"""
    return series.astype(str).str.replace('_', ' ', regex=False) \
                              .str.replace(r'\s+', ' ', regex=True) \
                              .str.strip().str.lower()

def load_model_and_data():
    """Load the trained model and reference data"""
    global model, descriptions, precautions, available_symptoms

    # Load model
    model = DiseasePredictionModel()
    try:
        model.load_model('models/disease_model.joblib')
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return False

    # Load and clean descriptions
    try:
        desc_df = pd.read_csv('data/symptom_Description.csv')
        desc_df['Disease'] = clean_text_column(desc_df['Disease'])
        descriptions.update(dict(zip(desc_df['Disease'], desc_df['Description'])))
    except FileNotFoundError:
        print("Description file not found.")

    # Load and clean precautions
    try:
        prec_df = pd.read_csv('data/symptom_precaution.csv')
        prec_df['Disease'] = clean_text_column(prec_df['Disease'])
        for _, row in prec_df.iterrows():
            disease = row['Disease']
            prec_list = [str(row[f'Precaution_{i}']) for i in range(1, 5) if pd.notna(row[f'Precaution_{i}'])]
            precautions[disease] = prec_list
    except FileNotFoundError:
        print("Precaution file not found.")

    # Get symptoms from model
    available_symptoms.extend(model.symptom_weights.keys())

    return True

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', symptoms=available_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease based on symptoms"""
    try:
        # Get symptoms from form
        symptoms = []
        for i in range(1, 18):  # Up to 17 symptoms
            symptom = request.form.get(f'symptom_{i}', '').strip().lower()
            if symptom:
                symptoms.append(symptom)

        if not symptoms:
            return jsonify({'error': 'Please select at least one symptom'})

        # Pad symptoms list to 17 elements
        while len(symptoms) < 17:
            symptoms.append('')

        # Make prediction
        predicted_disease, confidence = model.predict_disease(symptoms)

        # Get description and precautions
        description = descriptions.get(predicted_disease, "No description available")
        precaution_list = precautions.get(predicted_disease, ["No precautions available"])

        result = {
            'disease': predicted_disease,
            'confidence': f"{confidence:.2%}",
            'description': description,
            'precautions': precaution_list
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    if load_model_and_data():
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model and data. Please check your setup.")
