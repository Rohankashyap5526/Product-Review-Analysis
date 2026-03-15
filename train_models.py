"""
train_models.py
---------------
Trains Logistic Regression, Decision Tree, and Random Forest classifiers
and saves them to the models/ directory as .pkl files.
"""

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     confusion_matrix, classification_report)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import load_and_clean, encode_features


def evaluate(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    return {"name": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "cm": cm}


def train_all(data_path: str = "data/ecommerce_data.csv",
              models_dir: str = "models") -> list:
    os.makedirs(models_dir, exist_ok=True)

    print("Loading & cleaning data …")
    df = load_and_clean(data_path)

    print("Encoding features …")
    X, y, feature_names, encoders, scaler = encode_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Models ────────────────────────────────────────────────────────────────
    models_cfg = [
        ("Logistic Regression",
         LogisticRegression(max_iter=500, random_state=42)),
        ("Decision Tree",
         DecisionTreeClassifier(max_depth=8, random_state=42)),
        ("Random Forest",
         RandomForestClassifier(n_estimators=150, max_depth=10,
                                random_state=42, n_jobs=-1)),
    ]

    results = []
    trained_models = {}

    for name, clf in models_cfg:
        print(f"\nTraining {name} …")
        clf.fit(X_train, y_train)
        res = evaluate(name, clf, X_test, y_test)
        results.append(res)
        trained_models[name] = clf

        # Save individual model
        safe_name = name.lower().replace(" ","_")
        with open(f"{models_dir}/{safe_name}.pkl","wb") as f:
            pickle.dump(clf, f)
        print(f"  Saved → {models_dir}/{safe_name}.pkl")

    # ── Save shared artefacts ─────────────────────────────────────────────────
    artefacts = {
        "encoders":      encoders,
        "scaler":        scaler,
        "feature_names": feature_names,
        "X_test":        X_test,
        "y_test":        y_test,
        "results":       results,
    }
    with open(f"{models_dir}/artefacts.pkl","wb") as f:
        pickle.dump(artefacts, f)
    print(f"\nShared artefacts saved → {models_dir}/artefacts.pkl")

    # ── Feature importances (Random Forest) ──────────────────────────────────
    rf = trained_models["Random Forest"]
    fi = pd.DataFrame({
        "feature":    feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(f"{models_dir}/feature_importances.csv", index=False)
    print(f"Feature importances → {models_dir}/feature_importances.csv")

    return results


if __name__ == "__main__":
    train_all()
