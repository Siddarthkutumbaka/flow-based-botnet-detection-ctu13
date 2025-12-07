import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# =========================
# 1. LOAD & PREPROCESS DATA
# =========================

print("✅ Loading dataset...")
df = pd.read_csv("capture20110810.binetflow")

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

print("✅ Columns:")
print(df.columns.tolist())

# Convert label column to binary: 1 = botnet, 0 = benign
df["label"] = df["label"].apply(
    lambda x: 1 if "botnet" in str(x).lower() else 0
)

# Handle infinities / NaNs
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Keep only numeric columns (features + numeric label)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df = df[numeric_cols]

# Separate features and labels
y = df["label"]
X = df.drop(columns=["label"])

print("\n✅ Feature matrix shape:", X.shape)
print("✅ Label distribution:\n", y.value_counts())

# =========================
# 2. TRAIN/TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\n✅ Train shape:", X_train.shape)
print("✅ Test shape :", X_test.shape)

# Because GradientBoosting can be slow on 2.8M rows,
# we optionally subsample the training data for that model.
MAX_GB_SAMPLES = 300_000
if len(X_train) > MAX_GB_SAMPLES:
    X_train_gb, _, y_train_gb, _ = train_test_split(
        X_train,
        y_train,
        train_size=MAX_GB_SAMPLES,
        stratify=y_train,
        random_state=42,
    )
else:
    X_train_gb, y_train_gb = X_train, y_train

print(f"\n✅ GradientBoosting will train on {len(X_train_gb)} samples")

# =========================
# 3. EVALUATION HELPER
# =========================

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Get scores for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        try:
            y_score = model.decision_function(X_test)
        except AttributeError:
            y_score = y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_score)
    except ValueError:
        auc = float("nan")

    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }

# =========================
# 4. RANDOM FOREST
# =========================

print("\n🚀 Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)
rf_metrics = evaluate_model("Random Forest", rf, X_test, y_test)

# =========================
# 5. GRADIENT BOOSTED TREES (scikit-learn)
# =========================

print("\n🚀 Training Gradient Boosting (sklearn)...")
gb_clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
)

gb_clf.fit(X_train_gb, y_train_gb)
gb_metrics = evaluate_model("GradientBoosting", gb_clf, X_test, y_test)

# =========================
# 6. SUMMARY TABLE
# =========================

results = pd.DataFrame([rf_metrics, gb_metrics])
print("\n=== Summary ===")
print(results)

# =========================
# 7. FEATURE IMPORTANCE (RANDOM FOREST)
# =========================

import matplotlib.pyplot as plt

importances = rf.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\n=== Top 10 Feature Importances (Random Forest) ===")
print(importance_df.head(10))

# Plot feature importance
plt.figure()
plt.barh(
    importance_df["feature"].head(10)[::-1],
    importance_df["importance"].head(10)[::-1]
)
plt.xlabel("Importance Score")
plt.title("Top 10 Important Features (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# =========================
# 8. ROC CURVE PLOTTING
# =========================

from sklearn.metrics import roc_curve

# Random Forest ROC
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

# Gradient Boosting ROC
gb_probs = gb_clf.predict_proba(X_test)[:, 1]
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs)

plt.figure()
plt.plot(rf_fpr, rf_tpr, label="Random Forest")
plt.plot(gb_fpr, gb_tpr, label="Gradient Boosting")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Botnet Detection")
plt.legend()
plt.grid(True)
plt.savefig("roc_curves.png")
plt.close()

print("\n✅ ROC Curves saved as roc_curves.png")