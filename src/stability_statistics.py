import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

INPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "stability_results.csv"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "stability_statistical_test.csv"
)


def bootstrap_ci(values, n_bootstrap=5000, seed=42):
    """Calculate a 95% bootstrap confidence interval."""

    rng = np.random.default_rng(seed)
    values = np.asarray(values)

    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        bootstrap_means.append(
            np.mean(sample)
        )

    lower = np.percentile(
        bootstrap_means,
        2.5
    )

    upper = np.percentile(
        bootstrap_means,
        97.5
    )

    return lower, upper


def rank_biserial_effect(x, y):
    """
    Rank-biserial effect size for paired Wilcoxon comparison.
    Positive values indicate higher SHAP stability.
    """

    differences = np.asarray(x) - np.asarray(y)

    positive = np.sum(differences > 0)
    negative = np.sum(differences < 0)

    total = positive + negative

    if total == 0:
        return 0.0

    return (positive - negative) / total


def main():

    print("=" * 60)
    print("SHAP vs LIME STABILITY STATISTICAL ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "customer_index",
        "top_k",
        "shap_stability",
        "lime_stability"
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    results = []

    for top_k in sorted(df["top_k"].unique()):

        group = df[
            df["top_k"] == top_k
        ].copy()

        shap_values = (
            group["shap_stability"]
            .dropna()
            .values
        )

        lime_values = (
            group["lime_stability"]
            .dropna()
            .values
        )

        if len(shap_values) != len(lime_values):
            raise ValueError(
                f"Unequal sample sizes for Top-{top_k}"
            )

        difference = (
            shap_values - lime_values
        )

        mean_difference = np.mean(
            difference
        )

        median_difference = np.median(
            difference
        )

        ci_lower, ci_upper = bootstrap_ci(
            difference
        )

        # Paired non-parametric test.
        statistic, p_value = wilcoxon(
            shap_values,
            lime_values,
            alternative="two-sided"
        )

        effect_size = rank_biserial_effect(
            shap_values,
            lime_values
        )

        results.append({
            "top_k": int(top_k),
            "samples": len(shap_values),
            "mean_shap_stability": np.mean(
                shap_values
            ),
            "mean_lime_stability": np.mean(
                lime_values
            ),
            "mean_difference_shap_minus_lime":
                mean_difference,
            "median_difference":
                median_difference,
            "bootstrap_ci_lower":
                ci_lower,
            "bootstrap_ci_upper":
                ci_upper,
            "wilcoxon_statistic":
                statistic,
            "p_value":
                p_value,
            "rank_biserial_effect_size":
                effect_size,
            "significant_at_0.05":
                p_value < 0.05
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nResults:")

    for _, row in results_df.iterrows():

        print(
            f"\nTop-{int(row['top_k'])}"
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
            f"Mean difference: "
            f"{row['mean_difference_shap_minus_lime']:.4f}"
        )

        print(
            f"95% CI: "
            f"[{row['bootstrap_ci_lower']:.4f}, "
            f"{row['bootstrap_ci_upper']:.4f}]"
        )

        print(
            f"Wilcoxon p-value: "
            f"{row['p_value']:.6f}"
        )

        print(
            f"Rank-biserial effect size: "
            f"{row['rank_biserial_effect_size']:.4f}"
        )

        print(
            f"Significant at 0.05: "
            f"{row['significant_at_0.05']}"
        )

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 60)

    print(
        f"\nCreated:\n- {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()