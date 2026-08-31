# Student Productivity Pattern Analyzer 🎓

Analyze how factors like sleep, study time, extracurricular activities, and screen time relate to academic performance.

## 📊 Research Questions

1. Which habits correlate most with productivity?
2. Is studying longer always associated with better results?
3. Can we identify different "student profiles" using clustering?

## 🔍 Analyses Performed

- **Exploratory Data Analysis (EDA):** Distributions, missing values, summary stats
- **Correlation Analysis:** Which lifestyle factors correlate most with academic performance
- **Non-Linear Patterns:** Does more study time always = better grades?
- **Clustering (K-Means):** Identify distinct student profiles
- **SQL Analysis:** Group-based comparisons using pandasql

## 📁 Project Structure

```
student-productivity-analyzer/
├── data/                    # Raw dataset from Kaggle
├── scripts/
│   ├── 01_download_data.py  # Fetch data from Kaggle API
│   ├── 02_eda.py            # Exploratory Data Analysis
│   ├── 03_correlation.py    # Correlation & non-linear patterns
│   ├── 04_clustering.py     # K-Means student profiling
│   └── 05_sql_analysis.py   # SQL-style group analysis
├── outputs/                 # Generated plots and reports
├── notebooks/               # Jupyter notebooks (future)
├── requirements.txt
└── README.md
```

## 🛠️ Skills Used

- **Python** — pandas, matplotlib, seaborn, scikit-learn
- **Pandas** — data wrangling, groupby, aggregation
- **SQL (via pandasql)** — SQL queries on DataFrames
- **Clustering** — K-Means for student profiling

## 📥 Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure Kaggle API (get token from kaggle.com → Account → API)
mkdir ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json

# 3. Get the data (either option)
# Option A — real Kaggle dataset (needs API key)
python scripts/01_download_data.py
# Option B — synthetic fallback (no key needed, works immediately)
python scripts/generate_data.py

# 4. Run analyses
python scripts/02_eda.py
python scripts/03_correlation.py
python scripts/04_clustering.py
python scripts/05_sql_analysis.py
```

## 📈 Dataset

**Student Lifestyle & GPA Prediction Dataset** from Kaggle  
([source](https://www.kaggle.com/datasets/sarveshchhetri/student-lifestyle-vs-academic-performance-dataset))

- 8,000 student records
- 18 features including study habits, sleep, screen time, stress, and performance

> **Note:** The Kaggle dataset download requires a Kaggle API key (`kaggle.json`).
> If you don't have one, a synthetic dataset (`scripts/generate_data.py`) is included
> so all analyses can run immediately. To use the larger real Kaggle dataset, run
> `01_download_data.py` after configuring your key, then re-run the analyses.

## Key Findings

### 1. Which habits correlate most with productivity?

| Factor | Correlation with Final_Score |
|--------|------------------------------|
| Hours_Studied | **+0.70** (strongest) |
| Attendance | +0.35 |
| Sleep_Hours | +0.28 |
| Stress_Level | −0.27 |
| Screen_Time | −0.25 |

**Study hours dominate.** Sleep and attendance help; stress and screen time hurt.

### 2. Is studying longer always better?

Yes, but with diminishing returns. The **Top 10%** performers study ~21.5 hrs/week
vs ~9.5 hrs/week for the bottom 10%. But note: the top group also sleeps more (7.7h)
and has less screen time (3.3h). Performance is a *combination* — long hours alone,
without sleep/proper balance, doesn't produce top results.

### 3. Student profiles (K-Means clustering)

Two distinct clusters emerged:

| Profile | Students | Avg Study | Avg Sleep | Avg Screen | Avg Score |
|---------|----------|-----------|-----------|------------|-----------|
| **Low performer** | 511 | 12.4h | 6.8h | 4.4h | 54.9 |
| **High performer** | 489 | 17.9h | 7.4h | 3.6h | 70.7 |

High performers: study more, sleep more, use less screen time, lower stress.

## Visualizations (in `outputs/`)

- `02_distributions.png` / `02_categorical_counts.png` — data overview
- `03_correlation_heatmap.png` — correlation matrix
- `03_study_vs_score.png` — is more studying better?
- `03_lifestyle_vs_score.png` — sleep & screen vs performance
- `04_elbow_method.png` — optimal cluster count
- `04_pca_clusters.png` — clusters in 2D
- `04_cluster_comparison.png` / `04_cluster_profiles.csv` — profile summaries

---

*Built as a portfolio project demonstrating Python, Pandas, SQL, and data analysis skills.*
