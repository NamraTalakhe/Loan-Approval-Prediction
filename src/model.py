
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.data_preprocessing import (
    load_raw_data,
    prepare_features_target,
    build_preprocessor,
    PIPELINE_PATH
)

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

MODEL_PATH = os.path.join(BASE_DIR, 'model.joblib')

TEST_DIR = os.path.join(
    BASE_DIR, 'outputs', 'test_data'
)


def train_model():

    # Load original dataset
    df = load_raw_data()

    # Separate features and target
    X, y = prepare_features_target(df)

    # Split before preprocessing to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Dataset loaded successfully.")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Build and fit preprocessor only on training data
    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train Logistic Regression
    model = LogisticRegression(
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_processed, y_train)

    # Evaluate on unseen test data
    y_pred = model.predict(X_test_processed)

    print("\n--- Test Set Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save trained model and fitted preprocessor
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PIPELINE_PATH)

    # Save original test data for evaluation
    os.makedirs(TEST_DIR, exist_ok=True)

    X_test.to_csv(
        os.path.join(TEST_DIR, 'X_test.csv'),
        index=False
    )

    y_test.to_csv(
        os.path.join(TEST_DIR, 'y_test.csv'),
        index=False,
        header=['Loan_Status']
    )

    print("\nModel saved:", MODEL_PATH)
    print("Preprocessor saved:", PIPELINE_PATH)
    print("Test data saved:", TEST_DIR)


if __name__ == '__main__':
    train_model()
