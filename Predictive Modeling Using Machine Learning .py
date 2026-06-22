import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 6)

# ==========================================
# 1. GENERATE SYNTHETIC CHURN DATA
# ==========================================
print("--- Step 1: Generating Dataset ---")
np.random.seed(42)
n_samples = 1500

data = {
    "CustomerID": range(10000, 10000 + n_samples),
    "Age": np.random.randint(18, 70, size=n_samples),
    "Tenure_Months": np.random.randint(1, 60, size=n_samples),
    "Monthly_Charges": np.random.uniform(20.0, 120.0, size=n_samples),
    "Total_Charges": np.random.uniform(100.0, 5000.0, size=n_samples),
    "Contract_Type": np.random.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        size=n_samples,
        p=[0.5, 0.3, 0.2],
    ),
}

df = pd.DataFrame(data)

# Create a realistic target variable (Churn) based on rules + noise
# High monthly charges + Month-to-Month contract = higher chance to churn
churn_prob = (
    (df["Monthly_Charges"] / 120.0) * 0.4
    + (df["Contract_Type"] == "Month-to-Month") * 0.4
    + np.random.normal(0, 0.1, n_samples)
)
df["Churn"] = (churn_prob > 0.5).astype(int)

print(f"Dataset Shape: {df.shape}")
print(f"Class Distribution:\n{df['Churn'].value_counts(normalize=True)}\n")


# ==========================================
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# ==========================================
print("--- Step 2: Preprocessing & Feature Engineering ---")

# One-Hot Encoding for categorical features
df_encoded = pd.get_dummies(df, columns=["Contract_Type"], drop_first=True)

# Define Features (X) and Target (y)
X = df_encoded.drop(columns=["CustomerID", "Churn"])
y = df_encoded["Churn"]

# Train-Test Split (Stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling (Crucial for numerical uniformity)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 3. MODEL TRAINING
# ==========================================
print("--- Step 3: Training Random Forest Classifier ---")
model = RandomForestClassifier(
    n_estimators=100, max_depth=8, random_state=42, class_weight="balanced"
)
model.fit(X_train_scaled, y_train)


# ==========================================
# 4. EVALUATION & VISUALIZATION
# ==========================================
print("--- Step 4: Evaluating Model ---")

# Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Text Reports
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create folder for saving artifacts
os.makedirs("metrics_plots", exist_ok=True)

# Plot 1: Confusion Matrix
plt.figure()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("metrics_plots/confusion_matrix.png")
plt.close()

# Plot 2: ROC Curve
plt.figure()
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
plt.plot(
    fpr, tpr, label=f"Random Forest (AUC = {auc_score:.2f})", color="darkorange"
)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("metrics_plots/roc_curve.png")
plt.close()

# Plot 3: Feature Importance
plt.figure()
importances = model.feature_importances_
indices = np.argsort(importances)
plt.title("Feature Importances")
plt.barh(range(X.shape[1]), importances[indices], color="g", align="center")
plt.yticks(range(X.shape[1]), [X.columns[i] for i in indices])
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.savefig("metrics_plots/feature_importance.png")
plt.close()

print(
    "🎉 Model training complete. Evaluation plots saved to '/metrics_plots'!"
)
