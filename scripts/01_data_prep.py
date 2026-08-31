"""
Script 01: Data acquisition (no longer required).

The project ships with data/student_data.csv bundled in the repo, so you can
run everything immediately without any download or API key.

To regenerate / experiment with a fresh dataset, run:

    python scripts/generate_data.py

(For reference: the bundled data is a realistic synthetic recreation of the
"Student Lifestyle & GPA Prediction" style datasets found on Kaggle, so the
analysis pipeline works on real-world-shaped data without needing credentials.)
"""
import os
import sys


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")] if os.path.isdir(data_dir) else []

    if csv_files:
        print(f"Data already present in {data_dir}: {csv_files}")
        print("No download needed — the dataset is bundled in the repo.")
    else:
        print(f"No CSV found in {data_dir}. Generate one with:")
        print("    python scripts/generate_data.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
