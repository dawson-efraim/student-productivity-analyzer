"""
Script 05: SQL Analysis (using pandasql)
- Execute SQL queries on the DataFrame
- GROUP BY, HAVING, CASE, ranking, segmentation

Analytical questions addressed:
1. Performance by study-hours bracket (with HAVING to filter small groups)
2. Sleep quality segments: average score, study hours, screen time
3. Screen time segments: performance and study hours
4. Extracurricular participation comparison (using CASE to bucket)
5. Top 10% vs bottom 10% students: lifestyle differences (using PERCENT_RANK)
6. High stress vs low stress groups: compare all lifestyle factors
7. Multi-dimensional segment: "high study + low screen" vs rest (using HAVING)

All results are descriptive; no causal claims.
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
        raise FileNotFoundError(f"No CSV found in {DATA_DIR}. Run generate_data.py first.")
    return pd.read_csv(os.path.join(DATA_DIR, csv_files[0]))


def run_queries(df):
    """Run a series of SQL queries for analysis."""

    # Helper: run SQL with the df registered so FROM df works
    env = {"df": df}
    q = lambda sql: sqldf(sql, env)

    print("=" * 60)
    print("SQL ANALYSIS — Group Comparisons")
    print("=" * 60)

    # Detect columns
    study_col = "Hours_Studied" if "Hours_Studied" in df.columns else None
    score_col = "Final_Score" if "Final_Score" in df.columns else None
    sleep_col = "Sleep_Hours" if "Sleep_Hours" in df.columns else None
    screen_col = "Screen_Time" if "Screen_Time" in df.columns else None
    extra_col = "Extracurricular" if "Extracurricular" in df.columns else None
    stress_col = "Stress_Level" if "Stress_Level" in df.columns else None
    attend_col = "Attendance" if "Attendance" in df.columns else None

    if not all([study_col, score_col, sleep_col, screen_col, extra_col, stress_col, attend_col]):
        print(f"Missing columns: study={study_col}, score={score_col}, sleep={sleep_col}, screen={screen_col}, extra={extra_col}, stress={stress_col}, attend={attend_col}")
        return

    # 1. Study hours bracket with HAVING (filter groups with at least 20 students)
    print(f"\n--- Q1: Average {score_col} by {study_col} bracket (filter groups >= 20 students) ---")
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
        HAVING COUNT(*) >= 20
        ORDER BY avg_score DESC
    """)
    print(result.to_string(index=False))

    # 2. Sleep quality segments
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
            ROUND(AVG({study_col}), 2) AS avg_study_hours,
            ROUND(AVG({screen_col}), 2) AS avg_screen_time,
            ROUND(AVG({stress_col}), 2) AS avg_stress
        FROM df
        GROUP BY sleep_group
        ORDER BY avg_score DESC
    """)
    print(result.to_string(index=False))

    # 3. Screen time segments
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
            ROUND(AVG({study_col}), 2) AS avg_study_hours,
            ROUND(AVG({sleep_col}), 2) AS avg_sleep
        FROM df
        GROUP BY screen_group
        ORDER BY avg_score DESC
    """)
    print(result.to_string(index=False))

    # 4. Extracurricular: compare Yes vs No, with additional metrics
    print(f"\n--- Q4: {extra_col} participation vs Performance ---")
    result = q(f"""
        SELECT
            {extra_col} AS extracurricular,
            COUNT(*) AS students,
            ROUND(AVG({score_col}), 2) AS avg_score,
            ROUND(AVG({study_col}), 2) AS avg_study_hours,
            ROUND(AVG({sleep_col}), 2) AS avg_sleep,
            ROUND(AVG({screen_col}), 2) AS avg_screen_time,
            ROUND(AVG({stress_col}), 2) AS avg_stress
        FROM df
        GROUP BY {extra_col}
        ORDER BY avg_score DESC
    """)
    print(result.to_string(index=False))

    # 5. Top 10% vs Bottom 10% (using PERCENT_RANK)
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
            ROUND(AVG({stress_col}), 2) AS avg_stress,
            ROUND(AVG({attend_col}), 2) AS avg_attendance,
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

    # 6. High stress (>= 7) vs Low stress (<= 3)
    print(f"\n--- Q6: High stress ({stress_col} >= 7) vs Low stress ({stress_col} <= 3) ---")
    result = q(f"""
        SELECT
            CASE
                WHEN {stress_col} >= 7 THEN 'High stress'
                WHEN {stress_col} <= 3 THEN 'Low stress'
                ELSE 'Moderate stress'
            END AS stress_group,
            COUNT(*) AS students,
            ROUND(AVG({score_col}), 2) AS avg_score,
            ROUND(AVG({study_col}), 2) AS avg_study_hours,
            ROUND(AVG({sleep_col}), 2) AS avg_sleep,
            ROUND(AVG({screen_col}), 2) AS avg_screen_time,
            ROUND(AVG({attend_col}), 2) AS avg_attendance
        FROM df
        GROUP BY stress_group
        ORDER BY avg_score DESC
    """)
    print(result.to_string(index=False))

    # 7. Multi-dimensional: "High study (>= 20h) + Low screen (<= 3h)" vs rest
    print(f"\n--- Q7: High study (>=20h) & Low screen (<=3h) vs others ---")
    result = q(f"""
        SELECT
            CASE
                WHEN {study_col} >= 20 AND {screen_col} <= 3 THEN 'High study + Low screen'
                ELSE 'All others'
            END AS segment,
            COUNT(*) AS students,
            ROUND(AVG({score_col}), 2) AS avg_score,
            ROUND(AVG({sleep_col}), 2) AS avg_sleep,
            ROUND(AVG({stress_col}), 2) AS avg_stress
        FROM df
        GROUP BY segment
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