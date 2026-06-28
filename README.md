# Data Science Internship Project 3
## Unsupervised Learning (Customer Segmentation)
**DecodeLabs Internship 2026 - Amashi Fernando**

## Overview
This project builds a customer segmentation pipeline using unsupervised
learning to discover hidden behavioral groupings in retail customer
data. The focus is on **distance-based clustering done rigorously** —
scaling features correctly, compressing dimensionality with PCA,
mathematically justifying the number of clusters rather than guessing,
and translating abstract cluster centroids back into real, interpretable
business personas.

## Objectives
- Apply **Principal Component Analysis (PCA)** to reduce a 30-feature
  space down to a small number of interpretable dimensions
- Use the **Elbow Method** and **Silhouette Score** to mathematically
  justify the optimal number of K-Means clusters
- Reverse-engineer cluster centroids out of PCA/scaled space and back
  into original, human-readable units
- Translate the resulting clusters into actionable **business personas**

## Dataset
**Source:** [Customer Personality Analysis — Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

**Size:** 2,240 rows × 29 columns (2,236 rows after removing outliers)

| Column Group | Examples | Description |
|---|---|---|
| Demographics | Year_Birth, Education, Marital_Status, Income, Kidhome, Teenhome | Customer profile information |
| Engagement | Dt_Customer, Recency, NumWebVisitsMonth | Tenure and recent activity |
| Spend | MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds | Amount spent per product category |
| Purchase Channel | NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases | Where/how customers buy |
| Campaign Response | AcceptedCmp1–5, Response, Complain | Marketing campaign engagement history |

 This dataset's 29 raw columns
(expanded to 30 after feature engineering) gives PCA genuine work to do.

## Methodology

### 1. Data Cleaning
- Checked for missing values: **24 missing in `Income`**, imputed with
  the median (robust to outliers).
- Checked for duplicate rows: **0 found**.
- Fixed rare/typo category labels in `Marital_Status` (e.g. "Alone",
  "Absurd", "YOLO" → "Single").
- Removed implausible data-entry outliers: birth years before 1930 and
  incomes above $200,000.
- Dropped `Z_CostContact` and `Z_Revenue` — constant columns with zero
  variance that add no signal.
- **Final shape after cleaning: 2,236 rows × 27 columns.**

### 2. Feature Engineering
Engineered features were deliberately added to give PCA a genuinely
high-dimensional space to compress, per the project's "reduce 20+
columns" requirement:

| New Feature | Description |
|---|---|
| `Age` | 2014 − Year_Birth (2014 = dataset's reference/collection year) |
| `Customer_Tenure_Days` | Days since enrollment, relative to the most recent signup in the dataset |
| `Total_Spend` | Sum of spend across all 6 product categories |
| `Total_Purchases` | Sum of purchases across all 4 channels |
| `Total_Campaigns_Accepted` | Sum of all 5 campaign acceptances + final Response |
| `Total_Children` | Kidhome + Teenhome |
| `Avg_Spend_Per_Purchase` | Total_Spend / Total_Purchases (zero-safe) |
| `Education_Encoded` | Ordinal encoding (Basic → PhD) |
| `Marital_*` | One-hot encoded marital status dummies |

**Final feature count entering PCA: 30 numeric columns.**

### 3. Scale
All 30 features were standardized with `StandardScaler` (mean ≈ 0,
std ≈ 1) so that high-magnitude features like `Income` don't dominate
distance calculations over smaller-scale behavioral features.

### 4. Compress — PCA
- A full PCA fit showed **21 components were needed to retain 95% of
  total variance** — confirming the dataset is genuinely
  high-dimensional, unlike smaller toy datasets.
- For clustering and visualization, the data was reduced to **3
  principal components**, capturing **48.22%** of total variance:
  - **PC1 (30.53%)** — driven by `Total_Spend`, `Income`, `MntWines`,
    `NumCatalogPurchases`, `MntMeatProducts` → an overall
    *spending power* axis.
  - **PC2 (9.12%)** — driven by `Teenhome`, `NumDealsPurchases`,
    `Total_Campaigns_Accepted` (negative), `Total_Purchases` →
    a *household/deal-seeking* axis.
  - **PC3 (8.56%)** — driven by `Total_Campaigns_Accepted`, `Response`,
    `AcceptedCmp4`, `NumDealsPurchases` → a *campaign responsiveness*
    axis.

### 5. Cluster — Determining Optimal K
Both diagnostic methods were run across K = 2 to 10:

| K | WCSS | Silhouette |
|---|---|---|
| 2 | 16,445.8 | **0.4655** ← best silhouette |
| 3 | 11,921.3 | 0.4184 |
| 4 | 8,709.7 | 0.4317 ← elbow |
| 5 | 7,621.2 | 0.3591 |
| 6 | 6,793.0 | 0.3308 |
| 7 | 6,053.2 | 0.3345 |
| 8 | 5,437.1 | 0.3228 |
| 9 | 4,922.7 | 0.3321 |
| 10 | 4,477.4 | 0.3484 |

**Elbow and Silhouette disagreed** — a normal, expected outcome, not a
mistake. K=2 maximizes silhouette score but only separates customers
into two broad income-based blobs, which is too coarse to be a useful
segmentation. **K=4 (the elbow point) was chosen instead**, trading a
small amount of silhouette score (0.4317 vs 0.4655) for four distinct,
business-actionable personas.

### 6. Translate — Reverse-Engineering Centroids
K-Means was fit in PCA space, so its centroids are abstract coordinates
with no real-world meaning. These were inverse-transformed — first out
of PCA space, then out of scaled space — to recover centroids in
original units (actual income $, actual age, actual spend $), and
cross-checked against simple groupby means on the raw assigned data to
confirm consistency.

## Results

### Cluster Centroids (Original Units)
| Cluster | Age | Income | Total Spend | Campaigns Accepted | Children |
|---|---|---|---|---|---|
| 0 | 41.9 | $35,254 | $86 | 0.2 | 1.2 |
| 1 | 50.8 | $57,489 | $739 | 0.3 | 1.4 |
| 2 | 45.8 | $72,931 | $1,243 | 0.3 | 0.2 |
| 3 | 43.1 | $78,250 | $1,600 | 2.9 | 0.1 |

### Business Persona Summary
| Cluster | Size | % of Base | Persona | Suggested Action |
|---|---|---|---|---|
| 0 | 1,030 | 46.1% | **Budget-Conscious Families** — younger, lower income, low spend, more children, frequent web visits but few purchases | Value-driven promotions, deal alerts, low-cost product bundles |
| 1 | 582 | 26.0% | **Steady Mid-Tier Spenders** — older, moderate income, moderate spend, more children | Loyalty programs, family-oriented bundles, cross-sell offers |
| 2 | 453 | 20.3% | **Affluent Low-Engagement** — high income and spend, but low campaign response, no children | Premium product placement; campaigns need a different angle since past offers haven't landed |
| 3 | 171 | 7.6% | **High-Value Campaign Responders** — highest income and spend, by far the most campaign-responsive, no children | VIP/early-access programs — this is the smallest segment but the most marketing-responsive |

- **Final Silhouette Score (K=4): 0.4317**
- **Final dataset:** 2,236 rows after cleaning, 30 features used for clustering

## Repository Structure
```
DecodeLabs-Task3/
├── DecodeLabs_Task3.ipynb                        # Main analysis notebook
├── Dataset-marketing_campaign.csv                # Source dataset (tab-separated)
├── outputs/
│   ├── clustered_customers.csv                   # Full dataset with Cluster labels
│   ├── persona_summary.csv                       # Aggregated persona stats
│   └── cluster_centroids_original_scale.csv      # Reverse-engineered centroids
├── Visualizations/
│   ├── 01_pca_explained_variance.png             # 95% variance threshold chart
│   ├── 02_elbow_and_silhouette.png               # Elbow + Silhouette comparison
│   ├── 03_clusters_2d.png                        # PC1 vs PC2 scatter
│   ├── 04_clusters_3d.png                        # 3D PCA scatter
│   ├── 05_silhouette_plot.png                    # Per-sample silhouette diagnostic
README.md
```

## How to Run

**Option 1 — Google Colab**
1. Open `DecodeLabs_Task3.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload `Dataset-marketing_campaign.csv` to the Colab session
   (or mount Google Drive if stored there)
3. Run all cells: **Runtime → Run all**

Required libraries (pre-installed on Colab, except `kneed`):
```
pandas, numpy, matplotlib, seaborn, scikit-learn
```
```bash
pip install kneed
```

**Option 2 — VS Code**
1. Open the project folder in VS Code
2. Install the **Jupyter** extension (if not already installed)
3. Open `DecodeLabs_Task3.ipynb` and select your Python interpreter (e.g. Anaconda)
4. Make sure the dataset file is in the same folder as the notebook
5. Run all cells using the ▶️ **Run All** button at the top of the notebook

Install dependencies (if not using Anaconda's base environment):
```bash
pip install pandas numpy scikit-learn matplotlib seaborn kneed
```

## Tech Stack
- Python (Pandas, NumPy)
- Scikit-Learn (StandardScaler, PCA, KMeans, silhouette_score)
- Kneed (automated Elbow detection via the Kneedle algorithm)
- Matplotlib, Seaborn

## Conclusion
This project segmented 2,236 customers using a Scale → PCA → K-Means
pipeline across 30 engineered features. Full PCA showed 21 components
were needed to retain 95% of variance, confirming the dataset was
genuinely high-dimensional; 3 components were used for clustering,
capturing 48.2% of variance and loading mainly on spend/income, household
composition, and campaign responsiveness. The Elbow Method and Silhouette
Score disagreed (K=4 vs K=2) — K=2 scored higher (0.4655) but only split
customers by income into two broad groups, so K=4 was deliberately chosen
instead (silhouette 0.4317) to produce more actionable personas. Reverse-
engineering the centroids out of PCA/scaled space back into original
units produced four distinct, business-interpretable segments — from a
large low-spend group (46% of customers) to a small but highly
campaign-responsive high-value group (7.6% of customers, averaging
2.9 accepted campaigns versus 0.2–0.3 for every other segment).

## Author
Amashi Fernando — DecodeLabs Data Science Internship, June 10 - July 10 2026
