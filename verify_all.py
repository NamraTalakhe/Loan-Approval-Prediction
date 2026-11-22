
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
PYTHON = sys.executable

REQUIRED_FILES = [
    "data/raw/loan_data.csv",
    "model.joblib",
    "preprocessor.joblib",
    "outputs/evaluation_results.txt",
    "outputs/threshold_comparison.csv",
    "outputs/baseline_comparison.csv",
    "outputs/coefficient_table.csv",
    "outputs/confusion_matrix.csv",
    "outputs/decision_log.md",
    "outputs/predictions/predictions.csv",
    "outputs/predictions/example_prediction.csv",
    "outputs/plots/confusion_matrix.png",
    "outputs/plots/roc_curve.png",
    "outputs/plots/sigmoid_function.png",
    "outputs/plots/class_distribution.png",
    "outputs/plots/decision_behaviour.png",
    "outputs/plots/coefficient_importance.png",
    "outputs/plots/probability_distribution.png"
]


def run_step(description, command):
    print(f"\n{'=' * 55}")
    print(f"STEP: {description}")
    print(f"{'=' * 55}")

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)

        raise RuntimeError(
            f"Failed: {description}"
        )

    print(f"SUCCESS: {description}")


def main():
    print("LOAN APPROVAL PREDICTION - PROJECT VERIFICATION")

    # 1. Check original dataset
    dataset_path = os.path.join(
        ROOT, "data", "raw", "loan_data.csv"
    )

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    print("\nSUCCESS: Dataset found.")

    # 2. Train Logistic Regression model
    run_step(
        "Model Training",
        [PYTHON, "-m", "src.model"]
    )

    # 3. Run complete evaluation
    run_step(
        "Model Evaluation",
        [PYTHON, "-m", "src.evaluation"]
    )

    # 4. Verify Flask application
    run_step(
        "Flask Application Import",
        [
            PYTHON,
            "-c",
            "from app import app; print('Flask app loaded successfully')"
        ]
    )

    # 5. Check generated project files
    print("\n" + "=" * 55)
    print("CHECKING REQUIRED OUTPUT FILES")
    print("=" * 55)

    missing_files = []

    for relative_path in REQUIRED_FILES:
        full_path = os.path.join(ROOT, relative_path)

        if os.path.isfile(full_path):
            print(f"[PASS] {relative_path}")
        else:
            print(f"[MISSING] {relative_path}")
            missing_files.append(relative_path)

    if missing_files:
        raise FileNotFoundError(
            "Some required output files are missing."
        )

    print("\n" + "=" * 55)
    print("ALL PROJECT VERIFICATION CHECKS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nVERIFICATION FAILED: {error}")
        sys.exit(1)
