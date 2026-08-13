"""
Exploratory Data Analysis - Cardiotocography
Dataset: Cardiotocography (CTG.csv)

This script supports the analytical report by performing the required EDA,
preprocessing, data reduction, and transformation activities.

Activities included:
1. Load and describe the CTG data
2. Review data structure, missing values, duplicates, and summary statistics
3. Visualize quantitative and qualitative variables
4. Identify noisy/extreme values with the IQR method
5. Clean duplicate records
6. Perform data reduction using PCA and stratified sampling
7. Perform transformations: feature construction, aggregation, smoothing,
   normalization, and discretization

"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# 1. Setup and load data
# -----------------------------
DATA_PATH = Path("CTG.csv")
OUTPUT_DIR = Path("results/eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

ctg = pd.read_csv(DATA_PATH)
nsp_map = {1: "Normal", 2: "Suspect", 3: "Pathologic"}
ctg["NSP_Label"] = ctg["NSP"].map(nsp_map)

# -----------------------------
# 2. Structure and summary checks
# -----------------------------
structure_summary = {
    "original_rows": ctg.shape[0],
    "original_columns": ctg.shape[1],
    "total_missing_values": int(ctg.isnull().sum().sum()),
    "duplicate_rows": int(ctg.duplicated().sum()),
}

ctg.dtypes.astype(str).to_csv(OUTPUT_DIR / "data_types.csv", header=["dtype"])
ctg.describe().to_csv(OUTPUT_DIR / "summary_statistics_original.csv")
ctg.isnull().sum().to_csv(OUTPUT_DIR / "missing_values.csv", header=["missing_count"])
ctg["NSP_Label"].value_counts().rename("count").to_csv(OUTPUT_DIR / "class_distribution_original.csv")

# Clean duplicate records after documenting the original structure.
ctg_clean = ctg.drop_duplicates().copy()
structure_summary["cleaned_rows"] = ctg_clean.shape[0]
structure_summary["duplicates_removed"] = structure_summary["original_rows"] - structure_summary["cleaned_rows"]

class_counts_clean = ctg_clean["NSP_Label"].value_counts()
class_percent_clean = (ctg_clean["NSP_Label"].value_counts(normalize=True) * 100).round(2)
class_distribution_clean = pd.DataFrame({
    "count": class_counts_clean,
    "percent": class_percent_clean,
})
class_distribution_clean.to_csv(OUTPUT_DIR / "class_distribution_cleaned.csv")

# Selected summary statistics for report table.
selected_vars = ["LB", "ASTV", "ALTV", "Mean", "Variance"]
selected_summary = ctg_clean[selected_vars].agg(["mean", "std", "min", "median", "max"]).T.round(2)
selected_summary.to_csv(OUTPUT_DIR / "selected_summary_statistics_cleaned.csv")

# -----------------------------
# 3. Visualizations
# -----------------------------
# Figure 1: Class distribution
fig, ax = plt.subplots(figsize=(7, 5))
class_counts_clean.loc[["Normal", "Suspect", "Pathologic"]].plot(kind="bar", ax=ax)
ax.set_title("Fetal State Class Distribution After Cleaning")
ax.set_xlabel("Fetal State")
ax.set_ylabel("Count")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "figure_A1_class_distribution.png", dpi=150)
plt.close(fig)

# Figure 2: Histogram of baseline fetal heart rate
fig, ax = plt.subplots(figsize=(7, 5))
ctg_clean["LB"].hist(ax=ax, bins=30)
ax.set_title("Distribution of Baseline Fetal Heart Rate (LB)")
ax.set_xlabel("Baseline FHR (beats per minute)")
ax.set_ylabel("Frequency")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "figure_A2_hist_LB.png", dpi=150)
plt.close(fig)

# Figure 3: Boxplot of ASTV by fetal state
fig, ax = plt.subplots(figsize=(8, 5))
ctg_clean.boxplot(column="ASTV", by="NSP_Label", ax=ax)
ax.set_title("Abnormal Short-Term Variability (ASTV) by Fetal State")
ax.set_xlabel("Fetal State")
ax.set_ylabel("ASTV (%)")
plt.suptitle("")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "figure_A3_box_ASTV_by_NSP.png", dpi=150)
plt.close(fig)

# Numeric predictor columns: exclude target and 10-class label from predictor correlations/PCA.
numeric_cols = ctg_clean.select_dtypes(include=[np.number]).columns.tolist()
predictor_cols = [c for c in numeric_cols if c not in ["CLASS", "NSP"]]

# Figure 4: Correlation heatmap
corr = ctg_clean[predictor_cols].corr()
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, aspect="auto")
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)
fig.colorbar(im, ax=ax)
ax.set_title("Correlation Heatmap of CTG Numeric Predictors")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "figure_A4_correlation_heatmap.png", dpi=150)
plt.close(fig)

# Figure 5: Scatterplot for pairwise relationship
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(ctg_clean["LB"], ctg_clean["Mean"], alpha=0.5)
ax.set_title("Baseline FHR (LB) vs. Histogram Mean")
ax.set_xlabel("LB: Baseline FHR")
ax.set_ylabel("Mean: Histogram Mean")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "figure_A5_scatter_LB_vs_Mean.png", dpi=150)
plt.close(fig)

# -----------------------------
# 4. Data cleaning: outliers/noisy values
# -----------------------------
outlier_summary = []
for col in predictor_cols:
    q1 = ctg_clean[col].quantile(0.25)
    q3 = ctg_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((ctg_clean[col] < lower) | (ctg_clean[col] > upper)).sum()
    outlier_summary.append({
        "variable": col,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": int(outliers),
        "outlier_percent": round(outliers / len(ctg_clean) * 100, 2),
    })
outlier_df = pd.DataFrame(outlier_summary).sort_values("outlier_count", ascending=False)
outlier_df.to_csv(OUTPUT_DIR / "outlier_summary_iqr.csv", index=False)

# -----------------------------
# 5. Data transformations
# -----------------------------
# Feature construction: fetal heart rate histogram range
ctg_clean["FHR_Range"] = ctg_clean["Max"] - ctg_clean["Min"]

# Smoothing demonstration: winsorize ASTV at 1st and 99th percentiles.
# This produces a smoothed analysis version without deleting records.
astv_lower, astv_upper = ctg_clean["ASTV"].quantile([0.01, 0.99])
ctg_clean["ASTV_Smoothed"] = ctg_clean["ASTV"].clip(astv_lower, astv_upper)

# Aggregation: group-level means/medians by fetal state.
aggregation_summary = ctg_clean.groupby("NSP_Label")[["LB", "ASTV", "ALTV", "UC", "FHR_Range"]].agg(["mean", "median"]).round(2)
aggregation_summary.to_csv(OUTPUT_DIR / "aggregation_by_fetal_state.csv")

# Normalization example using Min-Max scaling.
scaler_minmax = MinMaxScaler()
normalized_cols = ["LB", "ASTV", "ALTV", "Mean", "Variance", "FHR_Range"]
ctg_normalized = ctg_clean.copy()
ctg_normalized[[f"{c}_norm" for c in normalized_cols]] = scaler_minmax.fit_transform(ctg_clean[normalized_cols])
ctg_normalized[["LB", "LB_norm", "ASTV", "ASTV_norm", "FHR_Range", "FHR_Range_norm"]].head(10).to_csv(
    OUTPUT_DIR / "normalization_example.csv", index=False
)

# Discretization example: create baseline FHR categories using clinical-style bins.
# These are analytical groupings, not clinical diagnoses.
ctg_clean["LB_Category"] = pd.cut(
    ctg_clean["LB"],
    bins=[0, 110, 160, np.inf],
    labels=["Low", "Typical", "High"],
    include_lowest=True,
)
ctg_clean["LB_Category"].value_counts(dropna=False).rename("count").to_csv(OUTPUT_DIR / "LB_category_counts.csv")

transformation_summary = pd.DataFrame({
    "activity": ["Feature construction", "Smoothing", "Aggregation", "Normalization", "Discretization"],
    "implementation": [
        "FHR_Range = Max - Min",
        "ASTV_Smoothed clips ASTV at the 1st and 99th percentiles",
        "Group mean and median of LB, ASTV, ALTV, UC, and FHR_Range by NSP_Label",
        "Min-Max scaled selected numeric variables to a 0-to-1 range",
        "LB_Category groups LB into Low, Typical, and High ranges",
    ],
})
transformation_summary.to_csv(OUTPUT_DIR / "transformation_summary.csv", index=False)

# -----------------------------
# 6. Data reduction: PCA and sampling
# -----------------------------
# Stratified sample by fetal state label, 20% from each class.
ctg_sample = (
    ctg_clean
    .groupby("NSP", group_keys=False)
    .sample(frac=0.20, random_state=42)
)
ctg_sample.to_csv(OUTPUT_DIR / "stratified_sample_20_percent.csv", index=False)

# PCA on standardized predictor variables.
scaler_standard = StandardScaler()
X_scaled = scaler_standard.fit_transform(ctg_clean[predictor_cols])
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained = pd.DataFrame({
    "principal_component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
})
explained.to_csv(OUTPUT_DIR / "pca_explained_variance.csv", index=False)

# Figure 6: PCA cumulative explained variance
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker="o")
ax.set_title("Cumulative Explained Variance by PCA Components")
ax.set_xlabel("Number of Principal Components")
ax.set_ylabel("Cumulative Explained Variance")
ax.set_ylim(0, 1.05)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "figure_A6_pca_cumulative_variance.png", dpi=150)
plt.close(fig)

# Figure 7: PCA scatter plot
pca_plot_df = ctg_clean.assign(PC1=X_pca[:, 0], PC2=X_pca[:, 1])
fig, ax = plt.subplots(figsize=(8, 6))
for label, group in pca_plot_df.groupby("NSP_Label"):
    ax.scatter(group["PC1"], group["PC2"], alpha=0.5, label=label)
ax.set_title("PCA Projection: First Two Principal Components")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "figure_A7_pca_scatter_pc1_pc2.png", dpi=150)
plt.close(fig)

# Save cleaned/transformed dataset.
ctg_clean.to_csv(OUTPUT_DIR / "ctg_clean_transformed.csv", index=False)

# -----------------------------
# 7. Key results summary
# -----------------------------
key_results = f"""
DATA 645 Unit 2 CTG EDA Key Results
===================================
Original dataset shape: {ctg.shape[0]} rows, {ctg.shape[1]} columns including NSP_Label added in script
Cleaned dataset shape: {ctg_clean.shape[0]} rows, {ctg_clean.shape[1]} columns after transformations
Total missing values: {structure_summary['total_missing_values']}
Duplicate rows removed: {structure_summary['duplicates_removed']}

Cleaned fetal state class distribution:
{class_distribution_clean.to_string()}

Selected summary statistics after duplicate removal:
{selected_summary.to_string()}

Top variables by IQR outlier count:
{outlier_df[['variable', 'outlier_count', 'outlier_percent']].head(10).to_string(index=False)}

PCA cumulative explained variance:
- First 1 component: {explained.loc[0, 'cumulative_explained_variance']:.4f}
- First 2 components: {explained.loc[1, 'cumulative_explained_variance']:.4f}
- First 3 components: {explained.loc[2, 'cumulative_explained_variance']:.4f}
- First 5 components: {explained.loc[4, 'cumulative_explained_variance']:.4f}
- First 10 components: {explained.loc[9, 'cumulative_explained_variance']:.4f}

Transformation activities completed:
{transformation_summary.to_string(index=False)}

Baseline FHR categories:
{ctg_clean['LB_Category'].value_counts(dropna=False).to_string()}
"""

with open(OUTPUT_DIR / "ctg_eda_key_results.txt", "w", encoding="utf-8") as f:
    f.write(key_results)

print(key_results)
print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
