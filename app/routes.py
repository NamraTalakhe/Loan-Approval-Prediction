
from flask import render_template, request, flash, redirect, url_for
import pandas as pd

from . import app
from src.utils import load_model_and_pipeline
from src.data_preprocessing import RAW_DATA_PATH


# Load model and preprocessing pipeline
model, preprocessor = load_model_and_pipeline()

# Read original feature columns
_raw_df = pd.read_csv(RAW_DATA_PATH, nrows=0)

FEATURE_COLUMNS = [
    col for col in _raw_df.columns
    if col not in ['Loan_ID', 'Loan_Status']
]

NUMERIC_COLUMNS = [
    'ApplicantIncome',
    'CoapplicantIncome',
    'LoanAmount',
    'Loan_Amount_Term',
    'Credit_History'
]

# Decision threshold
THRESHOLD = 0.5


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    input_data = {}
    errors = []

    # Collect and validate form inputs
    for col in FEATURE_COLUMNS:

        value = request.form.get(col)

        if value is None or value.strip() == '':
            errors.append(f'Missing value for {col}')
            continue

        if col in NUMERIC_COLUMNS:
            try:
                value = float(value)

                if value < 0:
                    errors.append(
                        f'{col} cannot be negative'
                    )
                    continue

            except ValueError:
                errors.append(
                    f'Invalid numeric value for {col}'
                )
                continue

        input_data[col] = value

    if errors:
        for error in errors:
            flash(error, 'danger')

        return redirect(url_for('index'))

    try:

        # Create input DataFrame
        df_input = pd.DataFrame(
            [input_data],
            columns=FEATURE_COLUMNS
        )

        # Apply fitted preprocessing pipeline
        X_processed = preprocessor.transform(df_input)

        # Get approval probability
        probabilities = model.predict_proba(X_processed)[0]

        approved_index = list(model.classes_).index(1)

        approval_probability = float(
            probabilities[approved_index]
        )

        # Apply decision threshold
        prediction = (
            1 if approval_probability >= THRESHOLD else 0
        )

        result = (
            'Approved' if prediction == 1 else 'Rejected'
        )

        probability_percent = (
            approval_probability * 100
        )

        category = (
            'success' if prediction == 1 else 'warning'
        )

        message = (
            f'Prediction Result: {result} | '
            f'Approval Probability: {probability_percent:.2f}% | '
            f'Decision Threshold: {THRESHOLD:.2f}'
        )

        flash(message, category)

    except Exception as error:

        app.logger.exception(
            'Prediction failed'
        )

        flash(
            'Unable to generate prediction. Please check the input values.',
            'danger'
        )

    return redirect(url_for('index'))
