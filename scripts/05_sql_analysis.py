"""
Script 05: SQL Analysis (using pandasql)
- Execute SQL queries on the DataFrame
- Group comparisons
- Ranked analysis
"""
import os
import pandas as pd
from pandasql import sqldf

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {DATA_DIR}. Run 01_download_data.py first.")
    return pd.read_csv(os.path.join(DATA_DIR, csv_files[0]))


def run_queries(df):
    """Run a series of SQL queries for analysis."""

    # Helper: run SQL with the df registered so FROM df works
    env = {"df": df}
    q = lambda sql: sqldf(sql, env)

    print("=" * 60)
    print("SQL ANALYSIS")
    print("=" * 60)

    # --- Query 1: Average performance by study hours bracket ---
    study_col = None
    score_col = None
    sleep_col = None
    screen_col = None
    extra_col = None

    for c in ["Hours_Studied"]:
        if c in df.columns:
            study_col = c
    for c in ["Final_Score", "Exam_Score"]:
        if c in df.columns:
            score_col = c
    for c in ["Sleep_Hours"]:
        if c in df.columns:
            sleep_col = c
    for c in ["Screen_Time"]:
        if c in df.columns:
            screen_col = c
    for c in ["Extracurricular"]:
        if c in df.columns:
            extra_col = c

    print(f"\nDetected columns: study={study_col}, score={score_col}, "
          f"sleep={sleep_col}, screen={screen_col}, extra={extra_col}")

    # 1. Study Hours bracket performance
    if study_col and score_col:
        print(f"\n--- Q1: Average {score_col} by {study_col} bracket ---")
        result = q(f"""
            SELECT
                CASE
                    WHEN {study_col} <= 2 THEN '0-2 hours'
                    WHEN {study_col} <= 4 THEN '2-4 hours'
                    WHEN {study_col} <= 6 THEN '4-6 hours'
                    WHEN {study_col} <= 8 THEN '6-8 hours'
                    ELSE '8+ hours'
                END AS study_bracket,
                COUNT(*) AS student_count,
                ROUND(AVG({score_col}), 2) AS avg_score,
                ROUND(MIN({score_col}), 2) AS min_score,
                ROUND(MAX({score_col}), 2) AS max_score,
                ROUND(AVG({sleep_col}), 2) AS avg_sleep,
                ROUND(AVG({screen_col}), 2) AS avg_screen_time
            FROM df
            GROUP BY study_bracket
            ORDER BY avg_score DESC
        """)
        print(result.to_string(index=False))

    # 2. Sleep quality impact
    if sleep_col and score_col:
        print(f"\n--- Q2: Performance by {sleep_col} range ---")
        result = q(f"""
            SELECT
                CASE
                    WHEN {sleep_col} <= 4 THEN 'Low sleep (<=4h)'
                    WHEN {sleep_col} <= 6 THEN 'Moderate sleep (4-6h)'
                    WHEN {sleep_col} <= 8 THEN 'Good sleep (6-8h)'
                    ELSE 'High sleep (8h+)'
                END AS sleep_group,
                COUNT(*) AS students,
                ROUND(AVG({score_col}), 2) AS avg_score,
                ROUND(AVG({study_col}), 2) AS avg_study_hours
            FROM df
            GROUP BY sleep_group
            ORDER BY avg_score DESC
        """)
        print(result.to_string(index=False))

    # 3. Screen time vs performance
    if screen_col and score_col:
        print(f"\n--- Q3: Performance by {screen_col} range ---")
        result = q(f"""
            SELECT
                CASE
                    WHEN {screen_col} <= 2 THEN 'Low screen (<=2h)'
                    WHEN {screen_col} <= 4 THEN 'Moderate (2-4h)'
                    WHEN {screen_col} <= 6 THEN 'High (4-6h)'
                    ELSE 'Very high (6h+)'
                END AS screen_group,
                COUNT(*) AS students,
                ROUND(AVG({score_col}), 2) AS avg_score,
                ROUND(AVG({study_col}), 2) AS avg_study_hours
            FROM df
            GROUP BY screen_group
            ORDER BY avg_score DESC
        """)
        print(result.to_string(index=False))

    # 4. Extracurricular impact
    if extra_col and score_col:
        print(f"\n--- Q4: {extra_col} vs Performance ---")
        result = q(f"""
            SELECT
                {extra_col} AS extracurricular,
                COUNT(*) AS students,
                ROUND(AVG({score_col}), 2) AS avg_score,
                ROUND(AVG({study_col}), 2) AS avg_study_hours,
                ROUND(AVG({sleep_col}), 2) AS avg_sleep
            FROM df
            GROUP BY {extra_col}
            ORDER BY avg_score DESC
        """)
        print(result.to_string(index=False))

    # 5. Top 10% students vs Bottom 10%
    if score_col:
        print(f"\n--- Q5: Top 10% vs Bottom 10% students (by {score_col}) ---")
        result = q(f"""
            SELECT
                CASE
                    WHEN rank_pct >= 0.9 THEN 'Top 10% (highest scores)'
                    WHEN rank_pct <= 0.1 THEN 'Bottom 10% (lowest scores)'
                    ELSE 'Middle 80%'
                END AS group_label,
                COUNT(*) AS students,
                ROUND(AVG({study_col}), 2) AS avg_study_hours,
                ROUND(AVG({sleep_col}), 2) AS avg_sleep,
                ROUND(AVG({screen_col}), 2) AS avg_screen_time,
                ROUND(AVG({score_col}), 2) AS avg_score
            FROM (
                SELECT *,
                    PERCENT_RANK() OVER (ORDER BY {score_col}) AS rank_pct
                FROM df
            )
            WHERE rank_pct <= 0.1 OR rank_pct >= 0.9
            GROUP BY group_label
            ORDER BY avg_score DESC
        """)
        print(result.to_string(index=False))


def main():
    df = load_data()
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    run_queries(df)
    print("\nSQL analysis complete.")


if __name__ == "__main__":
    main()
