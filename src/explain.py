

import numpy as np
import pandas as pd
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

TARGET = "DEFAULT"
TOP_K = 5
N_SAMPLES = 200

AGE_BINS = [0, 30, 45, 60, 100]
AGE_LABELS = ["Under 30", "30-45", "45-60", "60+"]


def jaccard(set_a, set_b):
    return len(set_a & set_b) / len(set_a | set_b)


def stratified_sample_by_age(X_test, n_samples):
    age_groups = pd.cut(X_test["AGE"], bins=AGE_BINS, labels=AGE_LABELS)
    per_group = max(1, n_samples // len(AGE_LABELS))
    parts = []
    for label in AGE_LABELS:
        group_df = X_test[age_groups == label]
        take = min(per_group, len(group_df))
        if take > 0:
            parts.append(group_df.sample(n=take, random_state=42))
    return pd.concat(parts).reset_index(drop=True)


def main():
    model = joblib.load("outputs/xgboost_model.pkl")
    feature_cols = joblib.load("outputs/feature_cols.pkl")

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols].reset_index(drop=True)

    sample = stratified_sample_by_age(X_test, N_SAMPLES)
    print(f"Sampled {len(sample)} customers (stratified by age group)")
    print(pd.cut(sample["AGE"], bins=AGE_BINS, labels=AGE_LABELS).value_counts())

    print("\nComputing SHAP values...")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150)
    plt.close()
    print("Saved outputs/shap_summary.png")

    print("Computing LIME explanations (this takes a bit longer)...")
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_cols,
        class_names=["No Default", "Default"],
        mode="classification",
        discretize_continuous=True,
    )

    def predict_fn(x):
        return model.predict_proba(pd.DataFrame(x, columns=feature_cols))

    rows = []
    for i in range(len(sample)):
        row = sample.iloc[i]
        shap_row = shap_values[i]
        shap_top = set(np.array(feature_cols)[np.argsort(-np.abs(shap_row))[:TOP_K]])

        lime_exp = explainer_lime.explain_instance(row.values, predict_fn, num_features=TOP_K)
        lime_top = set()
        for feat_str, _weight in lime_exp.as_list():
            for fname in feature_cols:
                if fname in feat_str:
                    lime_top.add(fname)
                    break

        agreement = jaccard(shap_top, lime_top)
        age = row["AGE"]
        rows.append({
            "customer_index": i, "age": age,
            "shap_top_features": ", ".join(shap_top),
            "lime_top_features": ", ".join(lime_top),
            "jaccard_agreement": agreement,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(sample)} customers explained")

    agreement_df = pd.DataFrame(rows)
    agreement_df.to_csv("outputs/shap_lime_agreement.csv", index=False)

    print("\n=== SHAP vs LIME Agreement Summary ===")
    print(f"Mean Jaccard agreement (top-{TOP_K} features): {agreement_df['jaccard_agreement'].mean():.3f}")
    print(f"Std dev: {agreement_df['jaccard_agreement'].std():.3f}")

    agreement_df["age_group"] = pd.cut(agreement_df["age"], bins=AGE_BINS, labels=AGE_LABELS)
    subgroup_summary = agreement_df.groupby("age_group", observed=True)["jaccard_agreement"].agg(["mean", "std", "count"])
    subgroup_summary.to_csv("outputs/agreement_by_age_group.csv")

    print("\n=== Agreement by Age Group (stratified sample) ===")
    print(subgroup_summary)

    plt.figure()
    agreement_df["jaccard_agreement"].hist(bins=10)
    plt.xlabel(f"Jaccard agreement (top-{TOP_K} features)")
    plt.ylabel("Number of customers")
    plt.title("SHAP vs LIME explanation agreement")
    plt.tight_layout()
    plt.savefig("outputs/agreement_histogram.png", dpi=150)
    plt.close()
    print("Saved outputs/agreement_histogram.png")


if __name__ == "__main__":
    main()