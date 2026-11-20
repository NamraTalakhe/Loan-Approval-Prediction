import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib

# Paths relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'loan_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model.joblib')

def load_processed_data():
    """Load the preprocessed CSV containing features and target column 'Loan_Status'."""
    return pd.read_csv(PROCESSED_DATA_PATH)

def train_and_evaluate(df: pd.DataFrame):
    """Train a Logistic Regression model and print evaluation metrics.

    Returns the trained model object.
    """
    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = LogisticRegression(solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("--- Evaluation metrics ---")
    print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"confusion_matrix: {confusion_matrix(y_test, y_pred).tolist()}")
    print(f"precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"f1_score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    return model

def main():
    df = load_processed_data()
    model = train_and_evaluate(df)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()
