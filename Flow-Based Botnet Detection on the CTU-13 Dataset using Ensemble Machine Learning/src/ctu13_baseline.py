import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/raw/capture20110810.binetflow"
SEED = 42
TEST_SIZE = 0.2

def ensure_dirs():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

def load_data(path):
    print("Loading dataset...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df["label"] = df["label"].apply(lambda x: 1 if "botnet" in str(x).lower() else 0)

    feature_cols = [c for c in df.columns if c != "label"]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)

    df = df.loc[:, df.nunique() > 1]
    df = df.loc[:, (df.abs().sum(axis=0) > 0)]

    print("Features after cleaning:", len(df.columns) - 1)
    return df

def safe_score(model, X):
    try:
        return model.predict_proba(X)[:, 1]
    except:
        return model.predict(X)

def evaluate(name, model, X_test, y_test):
    t0 = time.time()
    y_pred = model.predict(X_test)
    infer = time.time() - t0
    y_score = safe_score(model, X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_score)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "false_positive_rate": fpr,
        "inference_seconds": infer,
        "test_samples": len(X_test)
    }, y_score

def main():
    ensure_dirs()
    df = load_data(DATA_PATH)
    y = df["label"]
    X = df.drop(columns=["label"]).astype(np.float32).clip(-1e6, 1e6)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    lr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=3000, solver="saga", n_jobs=-1))
    ])

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED, class_weight="balanced_subsample")

    results = []
    roc_data = []

    for name, model in [("LogisticRegression", lr), ("RandomForest", rf)]:
        print("Training", name)
        model.fit(X_train, y_train)
        m, s = evaluate(name, model, X_test, y_test)
        results.append(m)
        roc_data.append((name, s))
        print(m)

    res = pd.DataFrame(results).sort_values("f1", ascending=False)
    res.to_csv("outputs/metrics/results.csv", index=False, lineterminator="\n")
    print(res)

    plt.figure()
    for name, s in roc_data:
        fpr, tpr, _ = roc_curve(y_test, s)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.savefig("outputs/figures/roc_curves.png")
    plt.close()

    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    imp[::-1].plot(kind="barh")
    plt.savefig("outputs/figures/feature_importance.png")
    plt.close()

if __name__ == "__main__":
    main()
