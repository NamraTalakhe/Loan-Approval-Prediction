
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

from src.data_preprocessing import (
    load_raw_data,
    prepare_features_target
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

MODEL_PATH = os.path.join(BASE_DIR, 'model.joblib')
PIPELINE_PATH = os.path.join(BASE_DIR, 'preprocessor.joblib')

TEST_DIR = os.path.join(BASE_DIR, 'outputs', 'test_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
PRED_DIR = os.path.join(OUTPUT_DIR, 'predictions')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics and confusion matrix."""

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else 0
    )

    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(
            y_true, y_pred, zero_division=0
        ),
        'Recall': recall_score(
            y_true, y_pred, zero_division=0
        ),
        'Specificity': specificity,
        'F1_Score': f1_score(
            y_true, y_pred, zero_division=0
        ),
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn
    }


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOTS_DIR, filename),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    print("Starting model evaluation...")

    # Load model and fitted preprocessor
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PIPELINE_PATH)

    # Load saved test data
    X_test = pd.read_csv(
        os.path.join(TEST_DIR, 'X_test.csv')
    )

    y_test = pd.read_csv(
        os.path.join(TEST_DIR, 'y_test.csv')
    )['Loan_Status'].values

    # Recreate training labels for baseline comparison
    raw_data = load_raw_data()
    X, y = prepare_features_target(raw_data)

    _, X_test_check, y_train, y_test_check = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    if not np.array_equal(y_test, y_test_check.values):
        raise ValueError(
            "Saved test labels do not match the original split."
        )

    # Transform test data only
    X_test_processed = preprocessor.transform(X_test)

    # Model predictions and probabilities
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    thresholds = [0.3, 0.5, 0.7]

    # --------------------------------------------------
    # THRESHOLD COMPARISON
    # --------------------------------------------------

    threshold_rows = []

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

        metrics = calculate_metrics(y_test, y_pred)
        metrics['Threshold'] = threshold

        threshold_rows.append(metrics)

    threshold_df = pd.DataFrame(threshold_rows)

    threshold_df.to_csv(
        os.path.join(OUTPUT_DIR, 'threshold_comparison.csv'),
        index=False
    )

    # Select threshold with highest test-set F1.
    # This is exploratory, not an independently validated optimum.
    best_index = threshold_df['F1_Score'].idxmax()
    recommended_threshold = float(
        threshold_df.loc[best_index, 'Threshold']
    )

    y_pred_default = (y_prob >= 0.5).astype(int)
    y_pred_selected = (
        y_prob >= recommended_threshold
    ).astype(int)

    default_metrics = calculate_metrics(
        y_test, y_pred_default
    )

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred_default,
        labels=[0, 1]
    ).ravel()

    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    cm_df = pd.DataFrame(
        cm,
        index=['Actual Rejected', 'Actual Approved'],
        columns=['Predicted Rejected', 'Predicted Approved']
    )

    cm_df.to_csv(
        os.path.join(OUTPUT_DIR, 'confusion_matrix.csv')
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm_df,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title('Confusion Matrix - Threshold 0.5')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')

    save_plot('confusion_matrix.png')

    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

    report = classification_report(
        y_test,
        y_pred_default,
        labels=[0, 1],
        target_names=['Rejected', 'Approved'],
        digits=4,
        zero_division=0
    )

    # --------------------------------------------------
    # ROC CURVE AND AUC
    # --------------------------------------------------

    roc_auc = roc_auc_score(y_test, y_prob)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(7, 5))

    plt.plot(
        fpr,
        tpr,
        label=f'Logistic Regression (AUC = {roc_auc:.3f})'
    )

    plt.plot(
        [0, 1],
        [0, 1],
        'k--',
        label='Random Classifier'
    )

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    save_plot('roc_curve.png')

    # --------------------------------------------------
    # MAJORITY CLASS BASELINE
    # --------------------------------------------------

    dummy = DummyClassifier(
        strategy='most_frequent'
    )

    dummy.fit(
        np.zeros((len(y_train), 1)),
        y_train
    )

    dummy_pred = dummy.predict(
        np.zeros((len(y_test), 1))
    )

    baseline_metrics = calculate_metrics(
        y_test, dummy_pred
    )

    baseline_df = pd.DataFrame({
        'Metric': [
            'Accuracy',
            'Precision',
            'Recall',
            'Specificity',
            'F1_Score'
        ],
        'LogisticRegression': [
            default_metrics['Accuracy'],
            default_metrics['Precision'],
            default_metrics['Recall'],
            default_metrics['Specificity'],
            default_metrics['F1_Score']
        ],
        'MajorityBaseline': [
            baseline_metrics['Accuracy'],
            baseline_metrics['Precision'],
            baseline_metrics['Recall'],
            baseline_metrics['Specificity'],
            baseline_metrics['F1_Score']
        ]
    })

    baseline_df.to_csv(
        os.path.join(OUTPUT_DIR, 'baseline_comparison.csv'),
        index=False
    )

    # --------------------------------------------------
    # PREDICTION CSV
    # --------------------------------------------------

    prediction_df = pd.DataFrame({
        'Actual_Status': y_test,
        'Approval_Probability': y_prob,
        'Predicted_At_0.3': (y_prob >= 0.3).astype(int),
        'Predicted_At_0.5': (y_prob >= 0.5).astype(int),
        'Predicted_At_0.7': (y_prob >= 0.7).astype(int),
        'Predicted_Selected_Threshold': y_pred_selected
    })

    prediction_df.to_csv(
        os.path.join(PRED_DIR, 'predictions.csv'),
        index=False
    )

    # --------------------------------------------------
    # COEFFICIENT ANALYSIS
    # --------------------------------------------------

    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_.flatten()

    coefficient_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    coefficient_df['Interpretation'] = np.where(
        coefficient_df['Coefficient'] > 0,
        'Increases log-odds of approval',
        np.where(
            coefficient_df['Coefficient'] < 0,
            'Decreases log-odds of approval',
            'No linear effect'
        )
    )

    coefficient_df.to_csv(
        os.path.join(OUTPUT_DIR, 'coefficient_table.csv'),
        index=False
    )

    intercept = float(model.intercept_[0])

    # Coefficient importance plot
    top_features = coefficient_df.reindex(
        coefficient_df['Coefficient'].abs()
        .sort_values(ascending=False)
        .index
    ).head(20)

    plt.figure(figsize=(9, 7))

    sns.barplot(
        data=top_features,
        x='Coefficient',
        y='Feature'
    )

    plt.title('Top 20 Logistic Regression Coefficients')
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')

    save_plot('coefficient_importance.png')

    # --------------------------------------------------
    # SIGMOID FUNCTION
    # --------------------------------------------------

    x_values = np.linspace(-10, 10, 400)
    sigmoid = 1 / (1 + np.exp(-x_values))

    plt.figure(figsize=(7, 5))
    plt.plot(x_values, sigmoid)

    plt.axhline(0.5, linestyle='--')
    plt.title('Sigmoid Function')
    plt.xlabel('Linear Model Output')
    plt.ylabel('Probability')
    plt.grid(True)

    save_plot('sigmoid_function.png')

    # --------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------

    class_counts = pd.Series(y).map({
        0: 'Rejected',
        1: 'Approved'
    }).value_counts()

    plt.figure(figsize=(7, 5))

    sns.barplot(
        x=class_counts.index,
        y=class_counts.values
    )

    plt.title('Loan Status Class Distribution')
    plt.xlabel('Loan Status')
    plt.ylabel('Number of Applications')

    save_plot('class_distribution.png')

       # --------------------------------------------------
    # PROBABILITY DISTRIBUTION
    # --------------------------------------------------

    probability_df = pd.DataFrame({
        'Approval_Probability': y_prob,
        'Actual_Status': pd.Series(y_test).map({
            0: 'Rejected',
            1: 'Approved'
        }).values
    })

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=probability_df,
        x='Approval_Probability',
        hue='Actual_Status',
        bins=20,
        stat='density',
        common_norm=False,
        element='step',
        fill=False
    )

    plt.title('Probability Distribution by Actual Loan Status')
    plt.xlabel('Predicted Approval Probability')
    plt.ylabel('Density')
   

    save_plot('probability_distribution.png')
    # --------------------------------------------------
    # EXAMPLE PREDICTION
    # --------------------------------------------------

    example_input = X_test.iloc[[0]]
    example_processed = preprocessor.transform(example_input)

    example_probability = model.predict_proba(
        example_processed
    )[0, 1]

    example_prediction = int(
        example_probability >= recommended_threshold
    )

    example_df = example_input.copy()
    example_df['Approval_Probability'] = example_probability
    example_df['Predicted_Status'] = (
        'Approved' if example_prediction == 1 else 'Rejected'
    )
    example_df['Decision_Threshold'] = recommended_threshold

    example_df.to_csv(
        os.path.join(PRED_DIR, 'example_prediction.csv'),
        index=False
    )

    # --------------------------------------------------
    # SAVE EVALUATION RESULTS
    # --------------------------------------------------

    with open(
        os.path.join(OUTPUT_DIR, 'evaluation_results.txt'),
        'w',
        encoding='utf-8'
    ) as file:

        file.write('LOAN APPROVAL PREDICTION - EVALUATION\n')
        file.write('=' * 45 + '\n\n')

        file.write('Dataset Information\n')
        file.write(f'Test samples: {len(y_test)}\n\n')

        file.write('Classification Metrics (Threshold = 0.5)\n')

        for metric, value in default_metrics.items():
            file.write(f'{metric}: {value}\n')

        file.write(f'ROC-AUC: {roc_auc:.4f}\n')
        file.write(f'Intercept: {intercept:.6f}\n')

        file.write('\nClassification Report:\n')
        file.write(report)

        file.write('\n\nRecommended Exploratory Threshold: ')
        file.write(f'{recommended_threshold}\n')

        file.write(
            'Recommendation is based on the highest F1 among '
            'the three tested thresholds on this test set. '
            'Independent validation is needed before deployment.\n'
        )

        file.write('\nExample Prediction:\n')
        file.write(
            f'Approval Probability: {example_probability:.4f}\n'
        )
        file.write(
            f'Predicted Status: {example_df["Predicted_Status"].iloc[0]}\n'
        )

    # --------------------------------------------------
    # DECISION LOG
    # --------------------------------------------------

    with open(
        os.path.join(OUTPUT_DIR, 'decision_log.md'),
        'w',
        encoding='utf-8'
    ) as file:

        file.write('# Decision Log\n\n')

        file.write('## Problem Framing\n')
        file.write(
            'Loan approval is a binary classification problem. '
            'Approved = 1 and Rejected = 0.\n\n'
        )

        file.write('## Data Preparation\n')
        file.write(
            'An 80:20 stratified train-test split was used. '
            'The preprocessing pipeline was fitted only on training data.\n\n'
        )

        file.write('## Model Choice\n')
        file.write(
            'Logistic Regression was selected because it provides '
            'probability estimates and interpretable coefficients.\n\n'
        )

        file.write('## Evaluation\n')
        file.write(
            'Accuracy, precision, recall, specificity, F1-score, '
            'confusion matrix and ROC-AUC were evaluated.\n\n'
        )

        file.write('## Threshold Choice\n')
        file.write(
            f'Thresholds 0.3, 0.5 and 0.7 were compared. '
            f'The highest test-set F1 occurred at {recommended_threshold}. '
            'This is exploratory and should be validated independently.\n\n'
        )

        file.write('## Limitations\n')
        file.write(
            '- Small dataset.\n'
            '- Class imbalance.\n'
            '- Possible historical data bias.\n'
            '- Logistic Regression has a linear decision boundary.\n'
            '- Test-set threshold selection may be optimistic.\n'
        )

    print("\nEvaluation completed successfully.")
    print("\nThreshold Comparison:")
    print(threshold_df.to_string(index=False))

    print("\nROC-AUC:", round(roc_auc, 4))
    print("Recommended exploratory threshold:", recommended_threshold)

    print("\nOutputs saved in:", OUTPUT_DIR)


if __name__ == '__main__':
    main()
