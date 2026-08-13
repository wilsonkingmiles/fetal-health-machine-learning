"""
Binary Logistic Regression - Cardiotocography
Dataset: UCI Cardiotocography (CTG.csv)
Objective: Predict whether a cardiotocography record is abnormal (suspect/pathologic)
           versus normal using binary logistic regression.

How to run:
1. Place CTG.csv in the same folder as this script, or update DATA_PATH below.
2. Run: python DATA645_Unit4_CTG_Logistic_Regression_script.py
3. Outputs will be saved to the unit4_logistic_outputs folder.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import statsmodels.api as sm

# -----------------------------
# Configuration
# -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, "CTG.csv")
DATA_PATH = DEFAULT_DATA_PATH
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "results", "logistic_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# -----------------------------
# 1. Load and inspect data
# -----------------------------
ctg = pd.read_csv(DATA_PATH)
original_shape = ctg.shape
missing_before = ctg.isna().sum()
duplicates_before = int(ctg.duplicated().sum())

# Remove exact duplicate rows as a general cleaning step.
ctg_clean = ctg.drop_duplicates().reset_index(drop=True)
clean_shape = ctg_clean.shape

# Binary target: Normal = 0, Abnormal = 1, where abnormal combines suspect and pathologic.
# Original NSP: 1 = normal, 2 = suspect, 3 = pathologic.
ctg_clean["Abnormal"] = (ctg_clean["NSP"] > 1).astype(int)
ctg_clean["Abnormal_Label"] = ctg_clean["Abnormal"].map({0: "Normal", 1: "Abnormal"})

# Exclude NSP because it is the original target.
# Exclude CLASS because it is another diagnostic class label and would create leakage.
feature_cols = [c for c in ctg_clean.columns if c not in ["NSP", "CLASS", "Abnormal", "Abnormal_Label"]]
X = ctg_clean[feature_cols]
y = ctg_clean["Abnormal"]

# Save structural summaries.
ctg_clean.describe().to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))
missing_before.to_csv(os.path.join(OUTPUT_DIR, "missing_values.csv"), header=["missing_count"])
ctg_clean["NSP"].value_counts().sort_index().to_csv(os.path.join(OUTPUT_DIR, "original_nsp_distribution.csv"), header=["count"])
y.value_counts().sort_index().to_csv(os.path.join(OUTPUT_DIR, "binary_target_distribution.csv"), header=["count"])

# -----------------------------
# 2. Exploratory visualizations
# -----------------------------
plt.figure(figsize=(7, 5))
ctg_clean["Abnormal_Label"].value_counts().loc[["Normal", "Abnormal"]].plot(kind="bar")
plt.title("Binary Target Distribution: Normal vs. Abnormal")
plt.xlabel("Fetal State")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_binary_target_distribution.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 5))
plt.hist(ctg_clean["LB"], bins=20, edgecolor="black")
plt.title("Distribution of Baseline Fetal Heart Rate (LB)")
plt.xlabel("LB: FHR Baseline (beats per minute)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_lb_histogram.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 5))
ctg_clean.boxplot(column="ASTV", by="Abnormal_Label")
plt.title("Abnormal Short-Term Variability by Fetal State")
plt.suptitle("")
plt.xlabel("Fetal State")
plt.ylabel("ASTV: % Time with Abnormal Short-Term Variability")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_astv_boxplot_by_target.png"), dpi=200)
plt.close()

# Correlations with the binary target.
corr_with_target = ctg_clean[feature_cols + ["Abnormal"]].corr(numeric_only=True)["Abnormal"].drop("Abnormal").sort_values(key=np.abs, ascending=False)
corr_with_target.to_csv(os.path.join(OUTPUT_DIR, "correlations_with_abnormal_target.csv"), header=["correlation"])

plt.figure(figsize=(8, 6))
top_corr = corr_with_target.head(10).sort_values()
top_corr.plot(kind="barh")
plt.title("Top 10 Feature Correlations with Abnormal Target")
plt.xlabel("Pearson Correlation with Abnormal Target")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_top_correlations_with_target.png"), dpi=200)
plt.close()

# -----------------------------
# 3. Preprocessing and train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

# Pipeline: standardization followed by logistic regression.
base_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000, solver="liblinear"))
])

# Parameter tuning. C is inverse regularization strength. Penalty controls L1 vs L2 regularization.
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l1", "l2"],
    "model__class_weight": [None, "balanced"]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(base_pipeline, param_grid=param_grid, cv=cv, scoring="f1", refit=True, n_jobs=1)
grid.fit(X_train, y_train)
model = grid.best_estimator_

# Predict probabilities on the holdout test set.
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate default threshold and a tuned threshold.
threshold_rows = []
for threshold in np.linspace(0.10, 0.90, 17):
    y_pred_t = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    threshold_rows.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred_t),
        "precision": precision_score(y_test, y_pred_t, zero_division=0),
        "recall_sensitivity": recall_score(y_test, y_pred_t, zero_division=0),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_test, y_pred_t, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    })
threshold_results = pd.DataFrame(threshold_rows)
threshold_results.to_csv(os.path.join(OUTPUT_DIR, "threshold_tuning_results.csv"), index=False)

# Choose the threshold with the highest F1 score for the abnormal class.
best_threshold = float(threshold_results.loc[threshold_results["f1"].idxmax(), "threshold"])
y_pred_default = (y_prob >= 0.50).astype(int)
y_pred_final = (y_prob >= best_threshold).astype(int)

# Metrics for final model.
def metric_dict(name, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "model_or_threshold": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_test, y_pred, zero_division=0),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

metrics = pd.DataFrame([
    metric_dict("default_threshold_0.50", y_pred_default),
    metric_dict(f"tuned_threshold_{best_threshold:.2f}", y_pred_final)
])
metrics.to_csv(os.path.join(OUTPUT_DIR, "model_performance_metrics.csv"), index=False)

pd.DataFrame(confusion_matrix(y_test, y_pred_final),
             index=["Actual Normal", "Actual Abnormal"],
             columns=["Predicted Normal", "Predicted Abnormal"]).to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix_final.csv"))

with open(os.path.join(OUTPUT_DIR, "classification_report_final.txt"), "w") as f:
    f.write(classification_report(y_test, y_pred_final, target_names=["Normal", "Abnormal"], digits=3))

# Confusion matrix plot.
cm = confusion_matrix(y_test, y_pred_final)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm)
ax.set_title(f"Confusion Matrix (Threshold = {best_threshold:.2f})")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Normal", "Abnormal"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Abnormal"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_confusion_matrix.png"), dpi=200)
plt.close()

# ROC curve.
fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
auc_value = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC AUC = {auc_value:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve for Abnormal CTG Classification")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate / Sensitivity")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig6_roc_curve.png"), dpi=200)
plt.close()

# Threshold tuning plot.
plt.figure(figsize=(7, 5))
plt.plot(threshold_results["threshold"], threshold_results["precision"], marker="o", label="Precision")
plt.plot(threshold_results["threshold"], threshold_results["recall_sensitivity"], marker="o", label="Recall/Sensitivity")
plt.plot(threshold_results["threshold"], threshold_results["f1"], marker="o", label="F1")
plt.axvline(best_threshold, linestyle="--", label=f"Selected Threshold = {best_threshold:.2f}")
plt.title("Threshold Tuning for Abnormal Class")
plt.xlabel("Probability Threshold")
plt.ylabel("Metric Value")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig7_threshold_tuning.png"), dpi=200)
plt.close()

# -----------------------------
# 4. Model properties with statsmodels GLM
# -----------------------------
# Fit a binomial GLM using standardized training features to summarize coefficients.
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_sm = sm.add_constant(X_train_scaled)
glm_result = sm.GLM(y_train, X_sm, family=sm.families.Binomial()).fit()

params = glm_result.params
conf = glm_result.conf_int()
coef_table = pd.DataFrame({
    "coefficient_log_odds": params,
    "odds_ratio": np.exp(params),
    "ci_lower_odds_ratio": np.exp(conf[0]),
    "ci_upper_odds_ratio": np.exp(conf[1]),
    "p_value": glm_result.pvalues
}).sort_values("p_value")
coef_table.to_csv(os.path.join(OUTPUT_DIR, "glm_coefficients_odds_ratios.csv"))

# Coefficient plot for top absolute coefficients excluding intercept.
coef_no_const = coef_table.drop(index="const", errors="ignore").copy()
coef_no_const["abs_coef"] = coef_no_const["coefficient_log_odds"].abs()
top_coef = coef_no_const.sort_values("abs_coef", ascending=False).head(10).sort_values("coefficient_log_odds")
plt.figure(figsize=(8, 6))
top_coef["coefficient_log_odds"].plot(kind="barh")
plt.title("Top Logistic Regression Coefficients (Standardized Features)")
plt.xlabel("Coefficient on Log-Odds Scale")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig8_top_coefficients.png"), dpi=200)
plt.close()

# Diagnostics: Pearson residuals vs fitted probabilities.
fitted_prob_train = glm_result.fittedvalues
pearson_resid = glm_result.resid_pearson
plt.figure(figsize=(7, 5))
plt.scatter(fitted_prob_train, pearson_resid, alpha=0.55)
plt.axhline(0, linestyle="--")
plt.title("Logistic Regression Diagnostic: Pearson Residuals vs Fitted Probabilities")
plt.xlabel("Fitted Probability of Abnormal CTG")
plt.ylabel("Pearson Residual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig9_residuals_vs_fitted.png"), dpi=200)
plt.close()

# Save a concise text summary.
with open(os.path.join(OUTPUT_DIR, "analysis_summary.txt"), "w") as f:
    f.write("DATA 645 Unit 4 Logistic Regression Analysis Summary\n")
    f.write("======================================================\n")
    f.write(f"Original dataset shape: {original_shape}\n")
    f.write(f"Cleaned dataset shape after duplicate removal: {clean_shape}\n")
    f.write(f"Missing values: {int(missing_before.sum())}\n")
    f.write(f"Duplicate rows removed: {duplicates_before}\n")
    f.write(f"Binary target distribution: {y.value_counts().to_dict()}\n")
    f.write(f"Feature columns used: {feature_cols}\n")
    f.write(f"Excluded columns: NSP target, CLASS diagnostic label to avoid leakage\n")
    f.write(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}\n")
    f.write(f"Best CV parameters: {grid.best_params_}\n")
    f.write(f"Best CV F1 score: {grid.best_score_:.4f}\n")
    f.write(f"ROC AUC on test set: {auc_value:.4f}\n")
    f.write(f"Selected threshold: {best_threshold:.2f}\n")
    f.write("\nFinal metrics using selected threshold:\n")
    f.write(metrics.to_string(index=False))
    f.write("\n\nGLM model properties:\n")
    f.write(f"AIC: {glm_result.aic:.3f}\n")
    f.write(f"Null deviance: {glm_result.null_deviance:.3f}\n")
    f.write(f"Residual deviance: {glm_result.deviance:.3f}\n")
    f.write(f"McFadden-like pseudo R2 (1 - deviance/null deviance): {1 - glm_result.deviance / glm_result.null_deviance:.3f}\n")
    f.write("\nTop coefficients by p-value:\n")
    f.write(coef_table.head(12).to_string())

print("Analysis complete.")
print(f"Outputs saved to: {OUTPUT_DIR}")
print(f"Best parameters: {grid.best_params_}")
print(f"Selected threshold: {best_threshold:.2f}")
print(metrics)
