"""
Script 02: Exploratory Data Analysis (EDA)
- Load dataset
- Summary statistics (central tendency, spread, shape)
- Missing values check
- IQR-based outlier screening
- Distribution plots for key variables
- Save plots to outputs/
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Display units for axis labels (values verified against generate_data.py)
UNITS = {
    "Hours_Studied": "hrs/week",
    "Sleep_Hours": "hrs/night",
    "Screen_Time": "hrs/day",
    "Attendance": "% of classes",
    "Stress_Level": "1-10 scale",
    "Final_Score": "points (0-100)",
}


def load_data():
    """Find and load the CSV from the data/ folder."""
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV found in {DATA_DIR}. Run generate_data.py first."
        )
    path = os.path.join(DATA_DIR, csv_files[0])
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    return df


def summary_stats(df):
    """Print summary statistics: shape, dtypes, missing, descriptives, shape, outliers."""
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW (n = %d records)" % len(df))
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")

    numeric = df.select_dtypes(include=["number"]).copy()
    numeric = numeric[[c for c in numeric.columns if "id" not in c.lower()]]
    print(f"\nNumeric Summary:\n{numeric.describe().round(2).to_string()}")

    # Distribution shape: skew (|skew| > 1 = highly skewed) and kurtosis
    shape = pd.DataFrame({
        "skew": numeric.skew().round(2),
        "kurtosis": numeric.kurt().round(2),
    })
    print(f"\nDistribution Shape (skew ~ 0 means symmetric):\n{shape.to_string()}")


def report_outliers(df):
    """IQR-based outlier screening per numeric column (1.5 * IQR rule)."""
    numeric = df.select_dtypes(include=["number"]).copy()
    numeric = numeric[[c for c in numeric.columns if "id" not in c.lower()]]
    print("\n--- Outlier screening (Tukey IQR rule: outside Q1 - 1.5*IQR / Q3 + 1.5*IQR) ---")
    for col in numeric.columns:
        q1, q3 = numeric[col].quantile(0.25), numeric[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((numeric[col] < lo) | (numeric[col] > hi)).sum()
        print(f"  {col:15s} bounds [{lo:6.2f}, {hi:6.2f}]  outliers: {n_out:4d} ({n_out/len(df):.1%})")
    print("  Note: mild outliers are expected from clipped normal distributions;")


def plot_distributions(df):
    """Plot distributions of numeric columns with units and context."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Skip ID-like columns
    numeric_cols = [c for c in numeric_cols if "id" not in c.lower()]

    n = len(numeric_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="steelblue", edgecolor="white")
        axes[i].set_title(col.replace("_", " "), fontsize=12, fontweight="bold")
        axes[i].set_xlabel(UNITS.get(col, ""))
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Distribution of Study & Lifestyle Variables (n={len(df):,}, synthetic data)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")


def plot_categorical(df):
    """Plot value counts for categorical columns."""
    cat_cols = df.select_dtypes(include=["string", "category", "object"]).columns.tolist()

    if not cat_cols:
        print("No categorical columns found.")
        return

    n = len(cat_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        counts = df[col].value_counts()
        counts.plot.bar(ax=axes[i], color="coral", edgecolor="white")
        axes[i].set_title(col.replace("_", " ") + " (counts)", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=0)
        for x, v in enumerate(counts.values):  # annotate counts on bars
            axes[i].text(x, v, str(v), ha="center", va="bottom", fontsize=10)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Variable Counts", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_categorical_counts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    df = load_data()
    summary_stats(df)
    report_outliers(df)
    plot_distributions(df)
    plot_categorical(df)
    print("\nEDA complete.")


if __name__ == "__main__":
    main()
