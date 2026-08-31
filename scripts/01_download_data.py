"""
Script 01: Download the Kaggle dataset
Dataset: Student Lifestyle & GPA Prediction Dataset
Source: https://www.kaggle.com/datasets/sarveshchhetri/student-lifestyle-vs-academic-performance-dataset
"""
import os
import subprocess
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATASET = "sarveshchhetri/student-lifestyle-vs-academic-performance-dataset"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Try kaggle CLI
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", DATA_DIR, "--unzip"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"Dataset downloaded to {DATA_DIR}")
        print(result.stdout)
    except FileNotFoundError:
        print("ERROR: kaggle CLI not found.")
        print("Install it: pip install kaggle")
        print(f"Then place kaggle.json in ~/.kaggle/")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Download failed:\n{e.stderr}")
        sys.exit(1)

    # Show downloaded files
    for f in os.listdir(DATA_DIR):
        fpath = os.path.join(DATA_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
