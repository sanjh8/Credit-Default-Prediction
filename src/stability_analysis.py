"""
Explanation Stability Analysis
==============================

Tests whether SHAP and LIME explanations remain stable after
small, realistic perturbations to financially meaningful features.

Uses the existing XGBoost model and the same 200 customers
used in the main SHAP-LIME analysis.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

TEST_FILE = os.path.join(DATA_DIR, "test.csv")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
MODEL_FILE = os.path.join(OUTPUT_DIR, "xgboost_model.pkl")
FEATURE_FILE = os.path.join(OUTPUT_DIR, "feature_cols.pkl")

AGREEMENT_FILE = os.path.join(
    OUTPUT_DIR, "shap_lime_agreement.csv"
)

RANDOM_SEED = 42
N_SAMPLES = 200
PERTURBATION = 0.05

TOP_K_VALUES = [3, 5, 10]

# Financially meaningful numerical features.
FINANCIAL_FEATURES = [
    "LIMIT_BAL",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def jaccard_similarity(set_a, set_b):
    """Calculate Jaccard similarity between two sets."""

    set_a = set(set_a)
    set_b = set(set_b)

    union = set_a | set_b

    if not union:
        return 1.0

    return len(set_a & set_b) / len(union)


def get_top_features_from_shap(model, X, feature_names, top_k=5):
    """Return top-k SHAP features for one observation."""

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    values = np.abs(shap_values[0])

    indices = np.argsort(values)[::-1][:top_k]

    return [
        feature_names[i]
        for i in indices
    ]


def get_lime_top_features(
    explainer,
    model,
    observation,
    feature_names,
    top_k=5
):
    """Return top-k LIME features."""

    explanation = explainer.explain_instance(
        observation,
        model.predict_proba,
        num_features=top_k
    )

    feature_indices = []

    for feature_description, weight in explanation.as_list():
        # LIME returns strings such as:
        # "PAY_0 <= 0.00"
        # "LIMIT_BAL > 1.00"
        #
        # Extract the feature by matching known feature names.
        matched = None

        for feature in feature_names:
            if feature in feature_description:
                matched = feature
                break

        if matched is not None:
            feature_indices.append(matched)

    return feature_indices[:top_k]


def create_perturbed_observation(
    observation,
    feature_names,
    rng
):
    """
    Create a realistic ±5% perturbation.

    Only financially meaningful numerical variables are changed.
    Values are kept non-negative.
    """

    perturbed = observation.copy()

    available = [
        feature
        for feature in FINANCIAL_FEATURES
        if feature in feature_names
    ]

    if not available:
        raise ValueError(
            "None of the expected financial features were found."
        )

    # Perturb several financial features rather than every feature.
    n_to_change = min(3, len(available))

    selected = rng.choice(
        available,
        size=n_to_change,
        replace=False
    )

    for feature in selected:

        index = feature_names.index(feature)

        original_value = perturbed[index]

        # Randomly choose +5% or -5%.
        direction = rng.choice([-1, 1])

        new_value = original_value * (
            1 + direction * PERTURBATION
        )

        # Keep monetary values non-negative.
        new_value = max(0.0, new_value)

        perturbed[index] = new_value

    return perturbed, list(selected)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 60)
    print("SHAP-LIME EXPLANATION STABILITY ANALYSIS")
    print("=" * 60)

    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading data and model...")

    test_df = pd.read_csv(TEST_FILE)
    train_df = pd.read_csv(TRAIN_FILE)

    model = joblib.load(MODEL_FILE)
    feature_cols = joblib.load(FEATURE_FILE)

    feature_cols = list(feature_cols)

    print(f"Test samples available: {len(test_df)}")
    print(f"Number of features: {len(feature_cols)}")

    # --------------------------------------------------------
    # Verify columns
    # --------------------------------------------------------

    missing = [
        col for col in feature_cols
        if col not in test_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing test columns: {missing}"
        )

    missing_train = [
        col for col in feature_cols
        if col not in train_df.columns
    ]

    if missing_train:
        raise ValueError(
            f"Missing training columns: {missing_train}"
        )

    # --------------------------------------------------------
    # Select 200 customers
    # --------------------------------------------------------

    n = min(N_SAMPLES, len(test_df))

    sample_df = test_df.iloc[:n].copy()

    X_train = train_df[feature_cols].values

    # --------------------------------------------------------
    # Create LIME explainer
    # --------------------------------------------------------

    lime_explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_cols,
        class_names=["No Default", "Default"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_SEED
    )

    # --------------------------------------------------------
    # SHAP explainer
    # --------------------------------------------------------

    shap_explainer = shap.TreeExplainer(model)

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    print("\nRunning stability analysis...")
    print(f"Samples: {n}")
    print(f"Perturbation magnitude: ±{PERTURBATION * 100:.0f}%")

    for sample_number in range(n):

        row = sample_df.iloc[sample_number]

        original = row[feature_cols].astype(float).values

        # ----------------------------------------------------
        # Perturb customer
        # ----------------------------------------------------

        perturbed, changed_features = (
            create_perturbed_observation(
                original,
                feature_cols,
                rng
            )
        )

        # ----------------------------------------------------
        # SHAP original
        # ----------------------------------------------------

        shap_original_values = (
            shap_explainer.shap_values(
                original.reshape(1, -1)
            )
        )

        if isinstance(shap_original_values, list):
            shap_original_values = shap_original_values[-1]

        shap_original_values = np.asarray(
            shap_original_values
        )

        if shap_original_values.ndim == 3:
            shap_original_values = (
                shap_original_values[:, :, -1]
            )

        shap_original_abs = np.abs(
            shap_original_values[0]
        )

        # ----------------------------------------------------
        # SHAP perturbed
        # ----------------------------------------------------

        shap_perturbed_values = (
            shap_explainer.shap_values(
                perturbed.reshape(1, -1)
            )
        )

        if isinstance(shap_perturbed_values, list):
            shap_perturbed_values = (
                shap_perturbed_values[-1]
            )

        shap_perturbed_values = np.asarray(
            shap_perturbed_values
        )

        if shap_perturbed_values.ndim == 3:
            shap_perturbed_values = (
                shap_perturbed_values[:, :, -1]
            )

        shap_perturbed_abs = np.abs(
            shap_perturbed_values[0]
        )

        # ----------------------------------------------------
        # LIME original
        # ----------------------------------------------------

        lime_original = get_lime_top_features(
            lime_explainer,
            model,
            original,
            feature_cols,
            top_k=10
        )

        # ----------------------------------------------------
        # LIME perturbed
        # ----------------------------------------------------

        lime_perturbed = get_lime_top_features(
            lime_explainer,
            model,
            perturbed,
            feature_cols,
            top_k=10
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        for top_k in TOP_K_VALUES:

            shap_original_top = [
                feature_cols[i]
                for i in np.argsort(
                    shap_original_abs
                )[::-1][:top_k]
            ]

            shap_perturbed_top = [
                feature_cols[i]
                for i in np.argsort(
                    shap_perturbed_abs
                )[::-1][:top_k]
            ]

            lime_original_top = (
                lime_original[:top_k]
            )

            lime_perturbed_top = (
                lime_perturbed[:top_k]
            )

            shap_stability = jaccard_similarity(
                shap_original_top,
                shap_perturbed_top
            )

            lime_stability = jaccard_similarity(
                lime_original_top,
                lime_perturbed_top
            )

            agreement_original = jaccard_similarity(
                shap_original_top,
                lime_original_top
            )

            agreement_perturbed = jaccard_similarity(
                shap_perturbed_top,
                lime_perturbed_top
            )

            results.append({
                "customer_index": sample_number,
                "top_k": top_k,
                "changed_features": "|".join(
                    changed_features
                ),
                "shap_stability": shap_stability,
                "lime_stability": lime_stability,
                "agreement_original": agreement_original,
                "agreement_perturbed": agreement_perturbed,
                "agreement_change": (
                    agreement_perturbed
                    - agreement_original
                )
            })

        if (sample_number + 1) % 25 == 0:
            print(
                f"Processed {sample_number + 1}/{n}"
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    stability_file = os.path.join(
        OUTPUT_DIR,
        "stability_results.csv"
    )

    results_df.to_csv(
        stability_file,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows = []

    for top_k in TOP_K_VALUES:

        group = results_df[
            results_df["top_k"] == top_k
        ]

        summary_rows.append({
            "top_k": top_k,
            "samples": len(group),
            "mean_shap_stability": group[
                "shap_stability"
            ].mean(),
            "mean_lime_stability": group[
                "lime_stability"
            ].mean(),
            "mean_original_agreement": group[
                "agreement_original"
            ].mean(),
            "mean_perturbed_agreement": group[
                "agreement_perturbed"
            ].mean(),
            "mean_agreement_change": group[
                "agreement_change"
            ].mean(),
            "std_shap_stability": group[
                "shap_stability"
            ].std(),
            "std_lime_stability": group[
                "lime_stability"
            ].std()
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_file = os.path.join(
        OUTPUT_DIR,
        "stability_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STABILITY RESULTS")
    print("=" * 60)

    for _, row in summary_df.iterrows():

        print(
            f"\nTop-{int(row['top_k'])}:"
        )

        print(
            f"SHAP stability: "
            f"{row['mean_shap_stability']:.4f}"
        )

        print(
            f"LIME stability: "
            f"{row['mean_lime_stability']:.4f}"
        )

        print(
            f"Original SHAP-LIME agreement: "
            f"{row['mean_original_agreement']:.4f}"
        )

        print(
            f"Perturbed SHAP-LIME agreement: "
            f"{row['mean_perturbed_agreement']:.4f}"
        )

        print(
            f"Agreement change: "
            f"{row['mean_agreement_change']:.4f}"
        )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    print("\nCreated:")
    print(f"- {stability_file}")
    print(f"- {summary_file}")


if __name__ == "__main__":
    main()