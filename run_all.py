import subprocess
import sys

steps = [
    "train_models.py",
    "time_series_analysis.py",
    "stream_pipeline.py",
]

for step in steps:
    print(f"\n=== Running {step} ===")
    subprocess.run([sys.executable, step], check=True)

print("\nAll offline analyses are complete.")
print("For the live dashboard run: streamlit run dashboard.py")
