"""
Script 03: Correlation Analysis
- Correlation heatmap (which habits correlate most with performance)
- Study hours vs Final Score (is more study always better?)
- Sleep vs Performance
- Screen Time vs Performance
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {DATA_DIR}. Run 01_download_data.py first.")
    return pd.read_csv(os.path.join(DATA_DIR, csv_files[0]))


def plot_correlation_heatmap(df):
    """Heatmap of correlations between all numeric variables."""
    numeric_df = df.select_dtypes(include=["number"]).copy()
    # Drop ID-like columns
    numeric_df = numeric_df[[c for c in numeric_df.columns if "id" not in c.lower()]]

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = pd.np.triu(pd.np.ones_like(corr, dtype=bool)) if hasattr(pd, "np") else None
    import numpy as np
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Print top correlations with Final_Score
    target = None
    for candidate in ["Final_Score", "Exam_Score", "GPA", "Score", "Academic_Performance"]:
        if candidate in corr.columns:
            target = candidate
            break

    if target:
        print(f"\n--- Top correlations with {target} ---")
        scores = corr[target].drop(target).sort_values(key=abs, ascending=False)
        for feat, val in scores.items():
            print(f"  {feat:35s} {val:+.3f}")

    return corr


def plot_study_vs_score(df):
    """Does studying longer always lead to better results?"""
    study_col = None
    score_col = None
    for c in ["Hours_Studied", "Hours_Studied_Per_Week", "study_hours_per_week"]:
        if c in df.columns:
            study_col = c
            break
    for c in ["Final_Score", "Exam_Score", "GPA", "Score"]:
        if c in df.columns:
            score_col = c
            break

    if not study_col or not score_col:
        print(f"Could not find study/score columns. Available: {list(df.columns)}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scatter plot
    axes[0].scatter(df[study_col], df[score_col], alpha=0.3, s=10, color="steelblue")
    axes[0].set_xlabel(study_col)
    axes[0].set_ylabel(score_col)
    axes[0].set_title(f"{study_col} vs {score_col}")

    # Add trend line
    z = pd.np.polyfit(df[study_col].dropna(), df[score_col].dropna(), 1) if hasattr(pd, "np") else None
    import numpy as np
    z = np.polyfit(df[study_col].dropna(), df[score_col].dropna(), 1)
    p = np.poly1d(z)
    x_range = np.linspace(df[study_col].min(), df[study_col].max(), 100)
    axes[0].plot(x_range, p(x_range), "r--", linewidth=2, label="Linear trend")
    axes[0].legend()

    # Binned averages
    df["study_bin"] = pd.cut(df[study_col], bins=8)
    bin_means = df.groupby("study_bin", observed=True)[score_col].mean()
    bin_means.plot(kind="bar", ax=axes[1], color="steelblue", edgecolor="white")
    axes[1].set_title(f"Average {score_col} by {study_col} Bin")
    axes[1].set_xlabel(f"{study_col} Range")
    axes[1].set_ylabel(f"Mean {score_col}")
    axes[1].tick_params(axis="x", rotation=45)
    df.drop(columns=["study_bin"], inplace=True)

    # Regression plot
    sns.regplot(data=df, x=study_col, y=score_col, scatter_kws={"alpha": 0.2, "s": 8},
                line_kws={"color": "red"}, ax=axes[2])
    axes[2].set_title(f"Regression: {study_col} → {score_col}")

    fig.suptitle("Is Studying Longer Always Better?", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_study_vs_score.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_lifestyle_vs_score(df):
    """Sleep and screen time vs performance."""
    score_col = None
    for c in ["Final_Score", "Exam_Score", "GPA", "Score"]:
        if c in df.columns:
            score_col = c
            break

    sleep_col = None
    for c in ["Sleep_Hours", "Sleep_Hours_Per_Day"]:
        if c in df.columns:
            sleep_col = c
            break

    screen_col = None
    for c in ["Screen_Time", "Social_Media_Hours"]:
        if c in df.columns:
            screen_col = c
            break

    if not all([score_col, sleep_col, screen_col]):
        print("Could not find sleep/screen/score columns.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.regplot(data=df, x=sleep_col, y=score_col, scatter_kws={"alpha": 0.2, "s": 8},
                line_kws={"color": "green"}, ax=axes[0])
    axes[0].set_title(f"{sleep_col} vs {score_col}")

    sns.regplot(data=df, x=screen_col, y=score_col, scatter_kws={"alpha": 0.2, "s": 8},
                line_kws={"color": "red"}, ax=axes[1])
    axes[1].set_title(f"{screen_col} vs {score_col}")

    fig.suptitle("Lifestyle Factors vs Academic Performance", fontsize=16, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_lifestyle_vs_score.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    df = load_data()
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    plot_correlation_heatmap(df)
    plot_study_vs_score(df)
    plot_lifestyle_vs_score(df)
    print("\nCorrelation analysis complete.")


if __name__ == "__main__":
    main()
