# Loan Approval Prediction using Logistic Regression

## 1. Overview

This project implements a **Loan Approval Prediction System** using **Multiple Logistic Regression**.

The system predicts whether a loan application is likely to be:

- **Approved (1)**
- **Rejected (0)**

The prediction is based on applicant information such as income, education, credit history, loan amount, and property area.

The project includes data preprocessing, model training, evaluation, probability prediction, and a Flask-based web application.

## 2. Dataset

The project uses the public Loan Prediction Dataset from Kaggle.
**Dataset Source:**
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

- Total records: 614
- Target column: `Loan_Status`
- Approved applications: `Y` mapped to 1
- Rejected applications: `N` mapped to 0
- `Loan_ID` is excluded from model training.

Dataset location:

`data/raw/loan_data.csv`

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Flask
- HTML and CSS
- Joblib

## 4. Machine Learning Workflow

1. Load the loan dataset.
2. Separate input features and target variable.
3. Split data into training and testing sets using an 80:20 ratio.
4. Use stratified sampling with `random_state=42`.
5. Handle missing numerical values using median imputation.
6. Handle missing categorical values using most-frequent imputation.
7. Apply One-Hot Encoding to categorical features.
8. Apply Standard Scaling to numerical features.
9. Train the Logistic Regression model.
10. Evaluate model performance using classification metrics and ROC-AUC.
11. Save the trained model and preprocessing pipeline.
12. Use Flask to generate predictions through a web interface.

Preprocessing is fitted only on training data to prevent data leakage.

## 5. Model Evaluation Results

The model was evaluated on 123 unseen test samples.

| Metric | Result |
|---|---:|
| Accuracy | 86.18% |
| Precision | 84.00% |
| Recall | 98.82% |
| F1 Score | 90.81% |
| Specificity | 57.89% |
| ROC-AUC | 85.23% |

### Confusion Matrix

| | Predicted Rejected | Predicted Approved |
|---|---:|---:|
| Actual Rejected | 22 | 16 |
| Actual Approved | 1 | 84 |

- True Positive (TP): 84
- True Negative (TN): 22
- False Positive (FP): 16
- False Negative (FN): 1

## 6. Threshold Comparison

The model was evaluated at three classification thresholds.

| Threshold | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| 0.3 | 86.18% | 84.00% | 98.82% | 90.81% |
| 0.5 | 86.18% | 84.00% | 98.82% | 90.81% |
| 0.7 | 81.30% | 86.90% | 85.88% | 86.39% |

The Flask application uses a decision threshold of **0.5**.

Thresholds 0.3 and 0.5 produced identical predictions on the test set. The threshold comparison is exploratory and requires independent validation before deployment.

## 7. Project Structure

```text
Loan-Approval-Prediction/
│
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── css/
│           └── styles.css
│
├── data/
│   └── raw/
│       └── loan_data.csv
│
├── outputs/
│   ├── plots/
│   ├── predictions/
│   ├── test_data/
│   ├── baseline_comparison.csv
│   ├── coefficient_table.csv
│   ├── confusion_matrix.csv
│   ├── decision_log.md
│   ├── evaluation_results.txt
│   └── threshold_comparison.csv
│
├── src/
│   ├── download_data.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   └── utils.py
│
├── model.joblib
├── preprocessor.joblib
├── requirements.txt
├── verify_all.py
└── README.md
```

## 8. Setup and Installation

Clone the repository:

```bash
git clone https://github.com/NamraTalakhe/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction
```

Create a virtual environment:

```bash
python -m venv env
```

Activate it on Windows PowerShell:

```powershell
.\env\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure that the dataset is available at:

```text
data/raw/loan_data.csv
```

## 9. Running the Project

### Train the Model

```bash
python -m src.model
```

### Run Model Evaluation

```bash
python -m src.evaluation
```

### Run Complete Project Verification

```bash
python verify_all.py
```

### Launch the Flask Web Application

```bash
python -c "from app import app; app.run(debug=True)"
```

Open the application in your browser:

http://127.0.0.1:5000

Enter applicant information and click the Predict button to view the predicted loan status and approval probability.

## 10. Generated Outputs

The project generates:

- Confusion matrix
- ROC curve
- Sigmoid function plot
- Class distribution plot
- Decision behaviour plot
- Probability distribution plot
- Coefficient importance plot
- Threshold comparison CSV
- Baseline comparison CSV
- Classification evaluation report
- Example prediction CSV
- Prediction results CSV
- Decision log

## 11. Limitations

- The dataset contains only 614 records.
- The dataset has an imbalanced target distribution.
- The model may reflect historical lending biases.
- Logistic Regression uses a linear decision boundary in the transformed feature space.
- Test-set threshold comparisons are exploratory.
- Predictions should not be treated as actual financial approval decisions.

## 12. Conclusion

The project successfully implements a Logistic Regression-based Loan Approval Prediction System.

The model achieved **86.18% accuracy** and an **ROC-AUC score of 0.8523** on the test dataset. The Flask application provides an interface to enter applicant details and view the predicted loan status with its approval probability.

## 13. References

- Kaggle Loan Prediction Dataset
- Scikit-learn Documentation: https://scikit-learn.org/
- Flask Documentation: https://flask.palletsprojects.com/
- Pandas Documentation: https://pandas.pydata.org/
