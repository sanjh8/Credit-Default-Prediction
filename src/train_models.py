

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)
from xgboost import XGBClassifier

TARGET = "DEFAULT"


def load_data():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    feature_cols = [c for c in train_df.columns if c != TARGET]
    return (
        train_df[feature_cols],
        train_df[TARGET],
        test_df[feature_cols],
        test_df[TARGET],
        feature_cols,
    )


def evaluate(name, model, X_test, y_test, results):
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    results.append(
        {
            "model": name,
            "AUC-ROC": roc_auc_score(y_test, proba),
            "Average Precision (PR-AUC)": average_precision_score(y_test, proba),
            "F1": f1_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
        }
    )
    print(f"{name}: AUC-ROC={results[-1]['AUC-ROC']:.4f}, F1={results[-1]['F1']:.4f}")


def main():
    X_train, y_train, X_test, y_test, feature_cols = load_data()
    results = []

    # --- Logistic Regression (needs scaled features) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    evaluate("Logistic Regression", logreg, X_test_scaled, y_test, results)
    joblib.dump({"model": logreg, "scaler": scaler}, "outputs/logreg.pkl")

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    evaluate("Random Forest", rf, X_test, y_test, results)
    joblib.dump(rf, "outputs/random_forest.pkl")

    # --- XGBoost (main model used for SHAP/LIME later) ---
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    evaluate("XGBoost", xgb, X_test, y_test, results)
    joblib.dump(xgb, "outputs/xgboost_model.pkl")

    # Save feature column order — explain.py needs this
    joblib.dump(feature_cols, "outputs/feature_cols.pkl")

    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/model_comparison.csv", index=False)
    print("\nSaved model comparison table to outputs/model_comparison.csv")
    print(results_df)


if __name__ == "__main__":
    main()
