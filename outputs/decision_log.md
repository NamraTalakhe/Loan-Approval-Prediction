# Decision Log

## Problem Framing
Loan approval is a binary classification problem. Approved = 1 and Rejected = 0.

## Data Preparation
An 80:20 stratified train-test split was used. The preprocessing pipeline was fitted only on training data.

## Model Choice
Logistic Regression was selected because it provides probability estimates and interpretable coefficients.

## Evaluation
Accuracy, precision, recall, specificity, F1-score, confusion matrix and ROC-AUC were evaluated.

## Threshold Choice
Thresholds 0.3, 0.5 and 0.7 were compared. The highest test-set F1 occurred at 0.3. This is exploratory and should be validated independently.

## Limitations
- Small dataset.
- Class imbalance.
- Possible historical data bias.
- Logistic Regression has a linear decision boundary.
- Test-set threshold selection may be optimistic.
