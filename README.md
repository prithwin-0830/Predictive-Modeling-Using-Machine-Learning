# Predictive-Modeling-Using-Machine-Learning
# Predictive Modeling: Customer Churn Classification

##  Project Overview
This repository contains an end-to-end Machine Learning pipeline designed to predict customer churn. Utilizing a structured classification workflow, the project processes behavioral data, engineers targeted features, and optimizes a Random Forest Classifier to preemptively flag churning users.

## ⚙️ Architecture & Pipeline
1. **Feature Engineering:** One-Hot Encoding for categorical components; feature scaling via `StandardScaler`.
2. **Data Stratification:** Evaluated via a split stratified by the target label to counteract class imbalances.
3. **Model Selection:** `RandomForestClassifier` selected for its ensemble robustness and native interpretability via feature significance tracking.
4. **Performance Metrics:** Evaluated comprehensively using Precision, Recall, F1-Score, Confusion Matrices, and ROC-AUC curves.

##  Required Packages
* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `seaborn`

##  Evaluation Artifacts
The script automatically exports the following performance visualizations to the `/metrics_plots` directory:
* `confusion_matrix.png`: To visualize type-I and type-II error frequencies.
* `roc_curve.png`: Quantifying the model’s true-positive vs. false-positive trade-off capability.
* `feature_importance.png`: Ranking the numerical weight and operational impact of each predictive metric.

##  Getting Started
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO.git](https://github.com/YOUR_USERNAME/YOUR_REPO.git)

# Install requirements
pip install -r requirements.txt

BY M PRITHWIN

# Execute the pipeline
python predictive_model.py
