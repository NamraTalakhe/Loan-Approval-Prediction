
import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Project paths
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

RAW_DATA_PATH = os.path.join(
    BASE_DIR, 'data', 'raw', 'loan_data.csv'
)

PIPELINE_PATH = os.path.join(
    BASE_DIR, 'preprocessor.joblib'
)


def load_raw_data():
    """Load the original loan dataset."""
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    return pd.read_csv(RAW_DATA_PATH)


def prepare_features_target(df):
    """Separate input features and target variable."""

    required_columns = ['Loan_ID', 'Loan_Status']

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    y = df['Loan_Status'].map({
        'Y': 1,
        'N': 0
    })

    if y.isna().any():
        raise ValueError(
            "Loan_Status contains missing or invalid values."
        )

    X = df.drop(
        columns=['Loan_ID', 'Loan_Status']
    )

    return X, y.astype(int)


def build_preprocessor(X):
    """Build preprocessing transformers using feature columns."""

    categorical_cols = X.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    numeric_cols = X.select_dtypes(
        include=['number']
    ).columns.tolist()

    categorical_pipeline = Pipeline([
        (
            'imputer',
            SimpleImputer(strategy='most_frequent')
        ),
        (
            'onehot',
            OneHotEncoder(handle_unknown='ignore')
        )
    ])

    numerical_pipeline = Pipeline([
        (
            'imputer',
            SimpleImputer(strategy='median')
        ),
        (
            'scaler',
            StandardScaler()
        )
    ])

    preprocessor = ColumnTransformer([
        (
            'cat',
            categorical_pipeline,
            categorical_cols
        ),
        (
            'num',
            numerical_pipeline,
            numeric_cols
        )
    ])

    return preprocessor
