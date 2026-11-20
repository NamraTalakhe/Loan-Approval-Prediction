import subprocess
import sys
import os

ROOT = os.path.abspath(os.path.dirname(__file__))
PY = sys.executable

def run(cmd, cwd=ROOT):
    print(f"\n--- RUNNING: {cmd} ---")
    result = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr)
        sys.exit(result.returncode)

# 1. Download dataset
run(f"{PY} -m src.download_data")

# 2. Preprocess data (includes EDA placeholders)
run(f"{PY} -m src.data_preprocessing")

# 3. Train model and show metrics
run(f"{PY} -m src.model")

# 4. Verify Flask app imports correctly (does not start server)
run(f"{PY} -c \"import app; print('Flask app loaded successfully')\"")

print("\n✅ All steps completed without errors.")
print("\nTo launch the web UI run:\n    python -m app")
