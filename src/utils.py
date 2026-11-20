import joblib
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model.joblib')
PIPELINE_PATH = os.path.join(PROJECT_ROOT, 'preprocessor.joblib')

def load_model_and_pipeline():
    """Load the trained LogisticRegression model and the preprocessing pipeline.
    Returns:
        model, pipeline
    """
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)
    return model, pipeline
