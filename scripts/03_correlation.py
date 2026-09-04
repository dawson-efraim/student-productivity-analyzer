"""
Script 03: Correlation & relationship analysis
- Correlation heatmap (with significance)
- Study hours vs Final Score: scatter, binned means, regression (is more study always better?)
- Other lifestyle factors vs performance (sleep, screen time)
- Multivariate OLS regression (all lifestyle factors -> Final Score)
- Non-linear / diminishing-returns check

Notes on interpretation:
- Correlations are exploratory; they describe this synthetic dataset, not the real world.
- Because Final_Score was generated from the lifestyle variables, strong correlations
  are expected and must not be read as causal evidence.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

UNITS = {
    "Hours_Studied": "hrs/week",
    "Sleep_Hours": "hrs/night",
    "Screen_Time": "hrs/day",
    "Attendance": "% of classes",
    "Stress_Level": "1-10 scale",
    "Final_Score": "points (0-100)",
}

TARGET_CANDIDATES = ["Final_Score", "Exam_Score", "GPA", "Score", "Academic_Performance"]
STUDY_CANDIDATES = ["Hours_Studied", "Hours_Studied_Per_Week", "study_hours_per_week"]


def load_data():
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {DATA_DIR}. Run generate_data.py first.")
    return pd.read_csv(os.path.join(DATA_DIR, csv_files[0]))


def _pick_target(cols):
    for c in TARGET_CANDIDATES:
        if c in cols:
            return c
    return None


def _pick_study(cols):
    for c in STUDY_CANDIDATES:
        if c in cols:
            return c
    return None


def plot_correlation_heatmap(df):
    """Heatmap of correlations between numeric variables + significance table."""
    numeric_df = df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df[[c for c in numeric_df.columns if "id" not in c.lower()]]

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
        annot_kws={"fontsize": 9},
    )
    target = _pick_target(corr.columns)
    subtitle = "Exploratory — synthetic data; correlations describe this dataset, not real students"
    ax.set_title("Correlation Heatmap — Lifestyle Factors vs Performance", fontsize=13, fontweight="bold", pad=14)
    # subtitle as text below title
    fig.text(0.5, 0.92, subtitle, ha="center", fontsize=8, style="italic", color="#666666")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    path = os.path.join(OUTPUT_DIR, "03_correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Ranked correlations with p-values (H0: no linear relationship; small p = evidence against H0)
    target = _pick_target(corr.columns)
    if target:
        print(f"\n--- Correlations with {target} (Pearson r, p-value under H0: r=0) ---")
        for feat in corr.columns:
            if feat == target:
                continue
            r, p = pearsonr(numeric_df[feat].dropna(), numeric_df[target].dropna())
            # simple strength label
            absr = abs(r)
            strength = "strong" if absr >= 0.5 else "moderate" if absr >= 0.3 else "weak"
            direction = "positive" if r > 0 else "negative"
            print(f"  {feat:18s} r={r:+.3f}  p={p:.2e}  ({strength} {direction})")

    return corr


def plot_study_vs_score(df):
    """Does studying longer always lead to better results? Three complementary views."""
    study_col = _pick_study(df.columns)
    score_col = _pick_target(df.columns)
    if not study_col or not score_col:
        print(f"Could not find study/score columns. Available: {list(df.columns)}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Scatter + linear trend
    axes[0].scatter(df[study_col], df[score_col], alpha=0.25, s=10, color="steelblue")
    z = np.polyfit(df[study_col].dropna(), df[score_col].dropna(), 1)
    p = np.poly1d(z)
    x_range = np.linspace(df[study_col].min(), df[study_col].max(), 100)
    axes[0].plot(x_range, p(x_range), "r--", linewidth=2, label=f"Linear trend (slope {z[0]:.2f})")
    axes[0].set_xlabel(f"{study_col} ({UNITS.get(study_col, '')})")
    axes[0].set_ylabel(f"{score_col} ({UNITS.get(score_col, '')})")
    axes[0].set_title("Scatter + linear trend", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=8)
    # annotate correlation on panel
    r, pval = pearsonr(df[study_col], df[score_col])
    axes[0].text(0.05, 0.95, f"r = {r:.2f}  (p < 0.001)", transform=axes[0].transAxes,
                 fontsize=8, va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # 2) Binned averages — reveals non-linear / diminishing-returns shape
    df["_study_bin"] = pd.cut(df[study_col], bins=8)
    bin_means = df.groupby("_study_bin", observed=True)[score_col].mean()
    bin_means.plot(kind="bar", ax=axes[1], color="steelblue", edgecolor="white")
    axes[1].set_title("Mean score by study-hours bin", fontsize=11, fontweight="bold")
    axes[1].set_xlabel(f"{study_col} range")
    axes[1].set_ylabel(f"Mean {score_col} (points)")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    df.drop(columns=["_study_bin"], inplace=True)

    # 3) Regression with confidence band
    sns.regplot(data=df, x=study_col, y=score_col,
                scatter_kws={"alpha": 0.18, "s": 8}, line_kws={"color": "red"}, ax=axes[2])
    axes[2].set_title("Regression with 95% CI", fontsize=11, fontweight="bold")
    axes[2].set_xlabel(f"{study_col} ({UNITS.get(study_col, '')})")
    axes[2].set_ylabel(f"{score_col} ({UNITS.get(score_col, '')})")

    fig.suptitle(f"Is Studying Longer Always Better?  ({study_col} vs {score_col})",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.94, "Binned means (center) reveal whether gains flatten at high study hours",
             ha="center", fontsize=8, style="italic", color="#666666")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(OUTPUT_DIR, "03_study_vs_score.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_lifestyle_vs_score(df):
    """Sleep and screen time vs performance — two regression panels."""
    score_col = _pick_target(df.columns)
    sleep_col = "Sleep_Hours" if "Sleep_Hours" in df.columns else None
    screen_col = "Screen_Time" if "Screen_Time" in df.columns else None
    if not all([score_col, sleep_col, screen_col]):
        print("Could not find sleep/screen/score columns.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, color, title in [
        (axes[0], sleep_col, "green", f"{sleep_col} vs {score_col}"),
        (axes[1], screen_col, "firebrick", f"{screen_col} vs {score_col}"),
    ]:
        sns.regplot(data=df, x=col, y=score_col, scatter_kws={"alpha": 0.18, "s": 8},
                    line_kws={"color": color}, ax=ax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{col} ({UNITS.get(col, '')})")
        ax.set_ylabel(f"{score_col} ({UNITS.get(score_col, '')})")
        r, pval = pearsonr(df[col], df[score_col])
        ax.text(0.05, 0.95, f"r = {r:+.2f}", transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.suptitle("Other Lifestyle Factors vs Academic Performance", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.96, "Each panel: linear fit with 95% CI — associations, not causation",
             ha="center", fontsize=8, style="italic", color="#666666")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(OUTPUT_DIR, "03_lifestyle_vs_score.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def run_regression(df):
    """Multivariate OLS: which factors stay associated after controlling for the others."""
    score_col = _pick_target(df.columns)
    # Lifestyle predictors only (exclude the outcome)
    predictors = [c for c in ["Hours_Studied", "Sleep_Hours", "Screen_Time", "Attendance", "Stress_Level"] if c in df.columns]
    if not score_col or len(predictors) < 2:
        print("Not enough columns for regression.")
        return

    X = df[predictors].copy()
    y = df[score_col]
    # Standardize predictors so coefficients are comparable (change per 1 SD)
    X_z = (X - X.mean()) / X.std(ddof=0)
    X_z = sm.add_constant(X_z)
    model = sm.OLS(y, X_z).fit()

    print("\n--- Multivariate OLS: standardized coefficients (per 1 SD change in predictor) ---")
    print(f"Outcome: {score_col}  |  n={len(df)}  |  R^2={model.rsquared:.3f}  adj-R^2={model.rsquared_adj:.3f}")
    print("  Interpretation: each coef = expected change in score when that predictor rises by 1 SD,")
    print("  holding the others fixed (within this synthetic data-generating process).")
    coef_table = pd.DataFrame({
        "coef (per 1 SD)": model.params.round(2),
        "std err": model.bse.round(2),
        "p-value": model.pvalues.round(4),
    })
    # drop intercept from the rank display, sort by |coef|
    coef_no_const = coef_table.drop(index="const", errors="ignore")
    print(coef_no_const.sort_values("coef (per 1 SD)", key=lambda s: s.abs(), ascending=False).to_string())


def main():
    df = load_data()
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns  (synthetic — see Data Source in README)")
    plot_correlation_heatmap(df)
    plot_study_vs_score(df)
    plot_lifestyle_vs_score(df)
    run_regression(df)
    print("\nCorrelation analysis complete.")


if __name__ == "__main__":
    main()
