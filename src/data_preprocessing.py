import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from sklearn.pipeline import Pipeline

# Paths relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'loan_data.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'loan_processed.csv')
PIPELINE_PATH = os.path.join(BASE_DIR, 'preprocessor.joblib')

def load_raw_data():
    """Load raw CSV file."""
    return pd.read_csv(RAW_DATA_PATH)

def preprocess(df: pd.DataFrame):
    """Clean, encode and scale the dataframe.
    Returns:
        X_processed (np.ndarray): transformed feature matrix
        y (pd.Series): target vector (0/1)
        preprocessor (ColumnTransformer): fitted transformer for later use
    """
    # Separate target
    y = df['Loan_Status'].map({'Y': 1, 'N': 0})
    X = df.drop(columns=['Loan_ID', 'Loan_Status'])

    # Identify column types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Pipelines


    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_pipe, categorical_cols),
            ('num', num_pipe, numeric_cols)
        ]
    )
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y, preprocessor

def save_processed(df: pd.DataFrame):
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

def main():
    raw = load_raw_data()
    X_proc, y, preprocessor = preprocess(raw)
    # Convert to DataFrame for optional saving
    if hasattr(X_proc, 'toarray'):
        X_arr = X_proc.toarray()
    else:
        X_arr = X_proc
    processed_df = pd.DataFrame(X_arr)
    processed_df['Loan_Status'] = y.values
    save_processed(processed_df)
    # Save the preprocessing pipeline for later use in model training and Flask app
    joblib.dump(preprocessor, PIPELINE_PATH)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")
    print(f"Preprocessor saved to {PIPELINE_PATH}")

if __name__ == '__main__':
    main()
