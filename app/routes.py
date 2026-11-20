from flask import render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np
from . import app
from src.utils import load_model_and_pipeline
from src.data_preprocessing import RAW_DATA_PATH

# Load model and preprocessing pipeline once at startup
model, preprocessor = load_model_and_pipeline()

# Determine original feature columns (without target) from raw CSV header
_raw_df = pd.read_csv(RAW_DATA_PATH, nrows=0)
FEATURE_COLUMNS = [col for col in _raw_df.columns if col not in ['Loan_ID', 'Loan_Status']]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    input_data = {}
    errors = []
    for col in FEATURE_COLUMNS:
        value = request.form.get(col)
        if value is None or value == '':
            errors.append(f"Missing value for {col}")
            continue
        # Convert numeric fields
        if col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
            try:
                # handle possible empty string after conversion
                value = float(value)
            except ValueError:
                errors.append(f"Invalid numeric value for {col}: {value}")
        input_data[col] = value
    if errors:
        for err in errors:
            flash(err, 'danger')
        return redirect(url_for('index'))

    # Build DataFrame matching training input order
    df_input = pd.DataFrame([input_data])
    # Apply same preprocessing
    X_processed = preprocessor.transform(df_input)
    # Predict
    pred = model.predict(X_processed)
    result = 'Approved' if pred[0] == 1 else 'Rejected'
    flash(f'Prediction result: {result}', 'success')
    return redirect(url_for('index'))
