"""
Script 02: Exploratory Data Analysis (EDA)
- Load dataset
- Summary statistics
- Missing values check
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


def load_data():
    """Find and load the CSV from the data/ folder."""
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV found in {DATA_DIR}. Run 01_download_data.py first."
        )
    path = os.path.join(DATA_DIR, csv_files[0])
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    return df


def summary_stats(df):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nNumeric Summary:\n{df.describe().round(2)}")


def plot_distributions(df):
    """Plot distributions of numeric columns."""
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
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("")
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Distribution of Numeric Variables", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")


def plot_categorical(df):
    """Plot value counts for categorical columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols:
        print("No categorical columns found.")
        return

    n = len(cat_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        df[col].value_counts().plot.bar(ax=axes[i], color="coral", edgecolor="white")
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].tick_params(axis="x", rotation=45)

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
    plot_distributions(df)
    plot_categorical(df)
    print("\nEDA complete.")


if __name__ == "__main__":
    main()
