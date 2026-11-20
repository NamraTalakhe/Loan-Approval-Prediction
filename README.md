# Loan Approval Prediction

This repository implements a **Loan Approval Prediction** system using **multiple logistic regression**.

### Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Pipeline](#running-the-pipeline)
- [Running the Web App](#running-the-web-app)
- [Testing](#testing)
- [References](#references)

### Overview
The goal is to predict whether a loan application will be **Approved** (`Y`) or **Rejected** (`N`) based on applicant information. The model is trained with **Logistic Regression** using multiple input features.

### Dataset
We use the public **Loan Prediction Dataset** from Kaggle:
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

The CSV (`loan_data.csv`) is automatically downloaded by the script and saved in `data/raw/`.

### Project Structure
```
Loan-Approval-Prediction/
├─ data/
│   ├─ raw/                  # original CSV (downloaded automatically)
│   └─ processed/            # cleaned CSV
├─ src/                      # data processing & model code
│   ├─ __init__.py
│   ├─ download_data.py
│   ├─ data_preprocessing.py
│   ├─ model.py
│   └─ utils.py
├─ app/                      # Flask web app
│   ├─ __init__.py
│   ├─ routes.py
│   ├─ templates/
│   │   └─ index.html
│   └─ static/css/
│       └─ styles.css
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ verify_all.py
├─ model.joblib            # trained Logistic Regression model (project root)
└─ preprocessor.joblib     # preprocessing pipeline (project root)
```

### Setup & Installation
```bash
# Clone the repo (or copy files to a folder)
python -m venv env
.
\env\Scripts\Activate.ps1   # PowerShell activation on Windows
pip install -r requirements.txt
```

### Running the Pipeline
```bash
python -m src.data_preprocessing   # creates data/processed/loan_processed.csv
python -m src.model                # trains model, prints metrics, saves model.joblib
```

### Running the Web App
```bash
python -c "from app import app; app.run(debug=True)"   # starts Flask server at http://127.0.0.1:5000
```
Open the URL in a browser, fill the form, and you will see **Approved** or **Rejected**.

### Testing
- **Dataset preprocessing** completed successfully and produced `data/processed/loan_processed.csv`.
- **Logistic Regression training** completed successfully with the following real metrics:
  - Accuracy: **86.18%**
  - Precision: **84.00%**
  - Recall: **98.82%**
  - F1 Score: **90.81%**
- **Approved prediction** tested successfully via the Flask UI.
- **Rejected prediction** tested successfully via the Flask UI.
- **Missing‑field validation** tested successfully; the app displays clear error alerts.

### References
- Kaggle Loan Prediction Dataset
- Scikit‑learn LogisticRegression documentation
- Flask documentation

---
*Feel free to modify the UI or model hyper‑parameters as needed.*
