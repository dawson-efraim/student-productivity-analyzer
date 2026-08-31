"""
Generate a realistic synthetic student dataset.

This dataset is bundled directly in the repo (data/student_data.csv) so the
project runs out-of-the-box — no Kaggle account or API key needed.

Columns:
  - Hours_Studied : weekly study hours (1-40)
  - Sleep_Hours   : average nightly sleep (3-11h)
  - Screen_Time   : daily non-study screen time (0-12h)
  - Attendance    : % of classes attended (70-100)
  - Extracurricular: participates in extracurriculars (Yes/No)
  - Stress_Level  : self-reported stress 1-10
  - Final_Score   : exam score 0-100

Final_Score is constructed from the lifestyle factors with injected noise,
so the data has realistic, learnable patterns (more study -> higher score,
more screen time -> lower score, etc).
"""
import pandas as pd
import numpy as np
import os

RNG_SEED = 42
N_SAMPLES = 1000

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_PATH = os.path.join(DATA_DIR, "student_data.csv")


def main():
    rng = np.random.default_rng(RNG_SEED)

    df = pd.DataFrame({
        "Hours_Studied": rng.normal(15, 5, N_SAMPLES).clip(1, 40),
        "Sleep_Hours": rng.normal(7, 1.2, N_SAMPLES).clip(3, 11),
        "Screen_Time": rng.normal(4, 2, N_SAMPLES).clip(0, 12),
        "Attendance": rng.uniform(70, 100, N_SAMPLES),
        "Extracurricular": rng.choice(["Yes", "No"], N_SAMPLES),
        "Stress_Level": rng.integers(1, 11, N_SAMPLES),
    })

    # Performance model: study/sleep/attendance help; screen/stress hurt
    df["Final_Score"] = (
        df["Hours_Studied"] * 1.5
        + df["Sleep_Hours"] * 2.5
        + df["Attendance"] * 0.4
        - df["Screen_Time"] * 1.5
        - df["Stress_Level"] * 1.0
        + rng.normal(0, 4, N_SAMPLES)
    ).clip(0, 100)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Dataset written: {OUT_PATH}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nPreview:")
    print(df.head().round(2).to_string())


if __name__ == "__main__":
    main()
