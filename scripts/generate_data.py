import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000
df = pd.DataFrame({
    "Hours_Studied": np.random.normal(15, 5, n).clip(1, 40),
    "Sleep_Hours": np.random.normal(7, 1.2, n).clip(3, 11),
    "Screen_Time": np.random.normal(4, 2, n).clip(0, 12),
    "Attendance": np.random.uniform(70, 100, n),
    "Extracurricular": np.random.choice(["Yes", "No"], n),
    "Stress_Level": np.random.randint(1, 11, n),
})
df["Final_Score"] = (
    df["Hours_Studied"] * 1.5 +
    df["Sleep_Hours"] * 2.5 +
    df["Attendance"] * 0.4 -
    df["Screen_Time"] * 1.5 -
    df["Stress_Level"] * 1.0 +
    np.random.normal(0, 4, n)
).clip(0, 100)
df.to_csv("C:/Users/dawso/Downloads/student-productivity-analyzer/data/student_data.csv", index=False)
print(f"Dataset generated: {len(df)} rows, {len(df.columns)} columns")
print(df.head())
