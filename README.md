# Fetal Health Machine Learning

End-to-end analysis of cardiotocography data using exploratory analysis and binary logistic regression.

## What this project demonstrates
- Python data analysis with pandas and NumPy
- Data quality checks and duplicate removal
- Exploratory visualization and correlation analysis
- PCA and feature transformation
- Train/test splitting with class stratification
- Standardization and logistic regression
- Hyperparameter and probability-threshold tuning
- Evaluation with precision, recall, F1, specificity, confusion matrices, and ROC AUC
- Awareness of class imbalance and target leakage

## Modeling approach
The original fetal-state outcome was converted to a binary target:
- **Normal = 0**
- **Abnormal = 1** (suspect or pathologic)

`CLASS` was excluded from the feature set to avoid target leakage.

## Selected results
At the default 0.50 threshold, the model achieved about **90.1% accuracy** and **0.958 ROC AUC**. Lowering the threshold to 0.30 increased abnormal-class recall from **0.75 to 0.843**, illustrating the tradeoff between sensitivity and false positives.

The saved metrics are available in `results/model_performance_metrics.csv`. The scripts generate the additional EDA and model visualizations locally when run.

## Repository structure
```text
scripts/
  exploratory_analysis.py
  logistic_regression.py
results/
  model_performance_metrics.csv
data/
  README.md
requirements.txt
```

## Run locally
1. Create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Place `CTG.csv` where the scripts expect it.
4. Run the exploratory analysis and logistic-regression scripts.

## Portfolio note
This repository is a cleaned portfolio presentation of academic work. The emphasis is on reproducible analysis, model evaluation, and clear technical communication. It is not intended for clinical use.
