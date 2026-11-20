import os
import requests

# Public CSV mirror on GitHub (no auth needed)
DATA_URL = "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)
DEST_PATH = os.path.join(RAW_DIR, 'loan_data.csv')

def download():
    print(f"Downloading dataset from {DATA_URL} ...")
    response = requests.get(DATA_URL)
    response.raise_for_status()
    with open(DEST_PATH, 'wb') as f:
        f.write(response.content)
    print(f"Dataset saved to {DEST_PATH}")

if __name__ == '__main__':
    download()
