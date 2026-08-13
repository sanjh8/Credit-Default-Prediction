

import numpy as np
import pandas as pd
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

AGE_BINS = [0, 30, 45, 60, 100]
AGE_LABELS = ["Under 30", "30-45", "45-60", "60+"]

N_SAMPLES = 200
MAX_TOP_K = 10


def jaccard(set_a, set_b):
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def stratified_sample_by_age(X_test, n_samples):
    age_groups = pd.cut(
        X_test["AGE"],
        bins=AGE_BINS,
        labels=AGE_LABELS
    )

    per_group = n_samples // len(AGE_LABELS)

    parts = []

    for label in AGE_LABELS:
        group_df = X_test[age_groups == label]

        take = min(per_group, len(group_df))

        if take > 0:
            parts.append(
                group_df.sample(
                    n=take,
                    random_state=42
                )
            )

    return pd.concat(parts).reset_index(drop=True)


def extract_lime_features(lime_exp, feature_cols, top_k):
    """
    Extract the actual feature names from LIME's explanation.
    Handles expressions such as:
    'PAY_0 <= 0.00'
    'BILL_AMT1 > 50000'
    """

    feature_names = []

    for feature_description, _weight in lime_exp.as_list():

        # LIME feature descriptions contain feature names.
        # Match longest feature names first to avoid partial matches.
        matched = None

        for fname in sorted(feature_cols, key=len, reverse=True):
            if fname in feature_description:
                matched = fname
                break

        if matched is not None and matched not in feature_names:
            feature_names.append(matched)

        if len(feature_names) >= top_k:
            break

    return feature_names


def main():

    print("=" * 60)
    print("GENERATING SHAP + LIME EXPLANATIONS")
    print("=" * 60)

    model = joblib.load(
        "outputs/xgboost_model.pkl"
    )

    feature_cols = joblib.load(
        "outputs/feature_cols.pkl"
    )

    train_df = pd.read_csv(
        "data/train.csv"
    )

    test_df = pd.read_csv(
        "data/test.csv"
    )

    X_train = train_df[feature_cols]

    X_test = test_df[
        feature_cols
    ].reset_index(drop=True)

    sample = stratified_sample_by_age(
        X_test,
        N_SAMPLES
    )

    print(
        f"\nSampled {len(sample)} customers "
        f"(age-stratified)"
    )

    print(
        pd.cut(
            sample["AGE"],
            bins=AGE_BINS,
            labels=AGE_LABELS
        ).value_counts()
    )

    # --------------------------------------------------
    # SHAP
    # --------------------------------------------------

    print("\nComputing SHAP values...")

    explainer_shap = shap.TreeExplainer(model)

    shap_values = explainer_shap.shap_values(
        sample
    )

    # SHAP summary plot
    plt.figure()

    shap.summary_plot(
        shap_values,
        sample,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/shap_summary.png",
        dpi=150
    )

    plt.close()

    print(
        "Saved outputs/shap_summary.png"
    )

    # --------------------------------------------------
    # LIME
    # --------------------------------------------------

    print(
        "\nInitializing LIME..."
    )

    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_cols,
        class_names=[
            "No Default",
            "Default"
        ],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    def predict_fn(x):

        return model.predict_proba(
            pd.DataFrame(
                x,
                columns=feature_cols
            )
        )

    # --------------------------------------------------
    # Generate explanations
    # --------------------------------------------------

    rows = []

    for i in range(len(sample)):

        row = sample.iloc[i]

        # -----------------------------
        # SHAP ranking
        # -----------------------------

        shap_row = shap_values[i]

        shap_ranked_indices = np.argsort(
            -np.abs(shap_row)
        )

        shap_features = [
            feature_cols[idx]
            for idx in shap_ranked_indices[
                :MAX_TOP_K
            ]
        ]

        # -----------------------------
        # LIME ranking
        # -----------------------------

        lime_exp = explainer_lime.explain_instance(
            row.values,
            predict_fn,
            num_features=MAX_TOP_K
        )

        lime_features = extract_lime_features(
            lime_exp,
            feature_cols,
            MAX_TOP_K
        )

        # -----------------------------
        # Agreement at K=3,5,10
        # -----------------------------

        shap_top3 = set(shap_features[:3])
        lime_top3 = set(lime_features[:3])

        shap_top5 = set(shap_features[:5])
        lime_top5 = set(lime_features[:5])

        shap_top10 = set(shap_features[:10])
        lime_top10 = set(lime_features[:10])

        rows.append({

            "customer_index": i,

            "age": row["AGE"],

            "shap_top3":
                ", ".join(shap_features[:3]),

            "lime_top3":
                ", ".join(lime_features[:3]),

            "jaccard_top3":
                jaccard(
                    shap_top3,
                    lime_top3
                ),

            "shap_top5":
                ", ".join(shap_features[:5]),

            "lime_top5":
                ", ".join(lime_features[:5]),

            "jaccard_top5":
                jaccard(
                    shap_top5,
                    lime_top5
                ),

            "shap_top10":
                ", ".join(shap_features[:10]),

            "lime_top10":
                ", ".join(lime_features[:10]),

            "jaccard_top10":
                jaccard(
                    shap_top10,
                    lime_top10
                ),

        })

        if (i + 1) % 20 == 0:

            print(
                f"  {i + 1}/{len(sample)} customers explained"
            )

    agreement_df = pd.DataFrame(rows)

    # --------------------------------------------------
    # Age groups
    # --------------------------------------------------

    agreement_df["age_group"] = pd.cut(
        agreement_df["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS
    )

    # --------------------------------------------------
    # Save main dataset
    # --------------------------------------------------

    agreement_df.to_csv(
        "outputs/shap_lime_agreement.csv",
        index=False
    )

    print(
        "\nSaved outputs/shap_lime_agreement.csv"
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print(
        "\n=== SHAP-LIME AGREEMENT ==="
    )

    for k in [3, 5, 10]:

        col = f"jaccard_top{k}"

        print(
            f"Top-{k}: "
            f"mean={agreement_df[col].mean():.4f}, "
            f"median={agreement_df[col].median():.4f}, "
            f"std={agreement_df[col].std():.4f}"
        )

    # --------------------------------------------------
    # Age-group analysis
    # --------------------------------------------------

    age_summary = (
        agreement_df
        .groupby(
            "age_group",
            observed=True
        )[
            "jaccard_top5"
        ]
        .agg(
            ["mean", "std", "count"]
        )
    )

    age_summary.to_csv(
        "outputs/agreement_by_age_group.csv"
    )

    print(
        "\n=== Agreement by Age Group ==="
    )

    print(age_summary)

    # --------------------------------------------------
    # Histogram
    # --------------------------------------------------

    plt.figure()

    agreement_df[
        "jaccard_top5"
    ].hist(bins=10)

    plt.xlabel(
        "Jaccard agreement (Top-5)"
    )

    plt.ylabel(
        "Number of customers"
    )

    plt.title(
        "SHAP vs LIME Explanation Agreement"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/agreement_histogram.png",
        dpi=150
    )

    plt.close()

    print(
        "\nSaved outputs/agreement_histogram.png"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "EXPLANATION GENERATION COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()