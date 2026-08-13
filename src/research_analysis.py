import numpy as np
import pandas as pd
from scipy.stats import kruskal
import matplotlib.pyplot as plt


INPUT = "outputs/shap_lime_agreement.csv"

AGE_BINS = [0, 30, 45, 60, 100]
AGE_LABELS = ["Under 30", "30-45", "45-60", "60+"]


def bootstrap_ci(data, n_boot=5000, seed=42):

    rng = np.random.default_rng(seed)

    data = np.asarray(data, dtype=float)

    means = []

    for _ in range(n_boot):

        sample = rng.choice(
            data,
            size=len(data),
            replace=True
        )

        means.append(sample.mean())

    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)

    return lower, upper


def main():

    print("=" * 60)
    print("SHAP-LIME RESEARCH ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(INPUT)

    required = [
        "jaccard_top3",
        "jaccard_top5",
        "jaccard_top10",
        "shap_top10",
        "lime_top10",
        "age",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------
    # Overall agreement
    # --------------------------------------------------

    agreement = (
        df["jaccard_top5"]
        .dropna()
        .astype(float)
    )

    mean = agreement.mean()
    median = agreement.median()
    std = agreement.std()

    ci_low, ci_high = bootstrap_ci(
        agreement
    )

    print("\n=== Overall SHAP-LIME Agreement ===")

    print(
        f"Samples: {len(agreement)}"
    )

    print(
        f"Mean: {mean:.4f}"
    )

    print(
        f"Median: {median:.4f}"
    )

    print(
        f"Standard deviation: {std:.4f}"
    )

    print(
        f"Minimum: {agreement.min():.4f}"
    )

    print(
        f"Maximum: {agreement.max():.4f}"
    )

    print(
        f"95% bootstrap CI: "
        f"[{ci_low:.4f}, {ci_high:.4f}]"
    )

    # --------------------------------------------------
    # Save overall statistics
    # --------------------------------------------------

    pd.DataFrame([{

        "overall_mean_jaccard": mean,

        "overall_median_jaccard": median,

        "overall_std": std,

        "bootstrap_ci_lower": ci_low,

        "bootstrap_ci_upper": ci_high,

    }]).to_csv(
        "outputs/research_statistics.csv",
        index=False
    )

    # --------------------------------------------------
    # Age groups
    # --------------------------------------------------

    df["age_group"] = pd.cut(
        df["age"],
        bins=AGE_BINS,
        labels=AGE_LABELS
    )

    print("\n=== Agreement by Age Group ===")

    age_results = []

    groups = []

    for age_group, group in df.groupby(
        "age_group",
        observed=True
    ):

        values = (
            group["jaccard_top5"]
            .dropna()
            .astype(float)
        )

        if len(values) == 0:
            continue

        low, high = bootstrap_ci(
            values
        )

        print(
            f"{age_group}: "
            f"mean={values.mean():.4f}, "
            f"median={values.median():.4f}, "
            f"std={values.std():.4f}, "
            f"95% CI=[{low:.4f}, {high:.4f}], "
            f"n={len(values)}"
        )

        age_results.append({

            "age_group": str(age_group),

            "mean": values.mean(),

            "median": values.median(),

            "std": values.std(),

            "count": len(values),

            "ci_lower": low,

            "ci_upper": high,

        })

        groups.append(values.values)

    age_results_df = pd.DataFrame(
        age_results
    )

    age_results_df.to_csv(
        "outputs/age_group_statistics.csv",
        index=False
    )

    # --------------------------------------------------
    # Kruskal-Wallis
    # --------------------------------------------------

    print("\n=== Kruskal-Wallis Test ===")

    if len(groups) >= 2:

        h_stat, p_value = kruskal(
            *groups
        )

        print(
            f"H-statistic: {h_stat:.4f}"
        )

        print(
            f"p-value: {p_value:.6f}"
        )

        if p_value < 0.05:

            print(
                "Result: Statistically significant "
                "difference detected between age groups."
            )

        else:

            print(
                "Result: No statistically significant "
                "difference detected between age groups."
            )

        # Epsilon squared
        n = len(df)
        k = len(groups)

        epsilon_squared = (
            h_stat - k + 1
        ) / (
            n - k
        )

        epsilon_squared = max(
            0,
            epsilon_squared
        )

        print(
            f"Epsilon-squared effect size: "
            f"{epsilon_squared:.4f}"
        )

        pd.DataFrame([{

            "H_statistic": h_stat,

            "p_value": p_value,

            "epsilon_squared": epsilon_squared,

            "significant_at_0.05": p_value < 0.05,

        }]).to_csv(
            "outputs/age_group_statistical_test.csv",
            index=False
        )

    # --------------------------------------------------
    # Top-K robustness
    # --------------------------------------------------

    print("\n=== Top-K Robustness Analysis ===")

    robustness = []

    for k in [3, 5, 10]:

        col = f"jaccard_top{k}"

        values = (
            df[col]
            .dropna()
            .astype(float)
        )

        low, high = bootstrap_ci(
            values
        )

        result = {

            "top_k": k,

            "samples": len(values),

            "mean_jaccard": values.mean(),

            "median_jaccard": values.median(),

            "std": values.std(),

            "bootstrap_ci_lower": low,

            "bootstrap_ci_upper": high,

        }

        robustness.append(result)

        print(
            f"Top-{k}: "
            f"mean={values.mean():.4f}, "
            f"median={values.median():.4f}, "
            f"95% CI=[{low:.4f}, {high:.4f}]"
        )

    robustness_df = pd.DataFrame(
        robustness
    )

    robustness_df.to_csv(
        "outputs/top_k_robustness.csv",
        index=False
    )

    # --------------------------------------------------
    # Plot Top-K robustness
    # --------------------------------------------------

    plt.figure()

    plt.plot(
        robustness_df["top_k"],
        robustness_df["mean_jaccard"],
        marker="o"
    )

    plt.xlabel(
        "Number of top features (K)"
    )

    plt.ylabel(
        "Mean Jaccard agreement"
    )

    plt.title(
        "SHAP-LIME Agreement Across Top-K Features"
    )

    plt.xticks([3, 5, 10])

    plt.tight_layout()

    plt.savefig(
        "outputs/top_k_robustness.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # Feature frequency
    # --------------------------------------------------

    print(
        "\n=== Feature-Level Frequency Analysis ==="
    )

    shap_counts = {}
    lime_counts = {}

    for features in df["shap_top10"].dropna():

        for feature in str(features).split(","):

            feature = feature.strip()

            if feature:

                shap_counts[feature] = (
                    shap_counts.get(feature, 0) + 1
                )

    for features in df["lime_top10"].dropna():

        for feature in str(features).split(","):

            feature = feature.strip()

            if feature:

                lime_counts[feature] = (
                    lime_counts.get(feature, 0) + 1
                )

    all_features = set(
        shap_counts
    ) | set(
        lime_counts
    )

    frequency_rows = []

    n_samples = len(df)

    for feature in all_features:

        shap_count = shap_counts.get(
            feature,
            0
        )

        lime_count = lime_counts.get(
            feature,
            0
        )

        frequency_rows.append({

            "feature": feature,

            "shap_count": shap_count,

            "lime_count": lime_count,

            "shap_frequency":
                shap_count / n_samples,

            "lime_frequency":
                lime_count / n_samples,

            "combined_frequency":
                (
                    shap_count + lime_count
                ) / (2 * n_samples),

        })

    frequency_df = pd.DataFrame(
        frequency_rows
    )

    frequency_df = frequency_df.sort_values(
        "combined_frequency",
        ascending=False
    )

    frequency_df.to_csv(
        "outputs/feature_frequency_comparison.csv",
        index=False
    )

    print(
        "\nTop 15 features by combined "
        "SHAP/LIME frequency:"
    )

    print(
        frequency_df.head(15).to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    pd.DataFrame([{

        "overall_mean_jaccard":
            mean,

        "overall_median_jaccard":
            median,

        "overall_std":
            std,

        "bootstrap_ci_lower":
            ci_low,

        "bootstrap_ci_upper":
            ci_high,

    }]).to_csv(
        "outputs/final_research_summary.csv",
        index=False
    )

    print("\n" + "=" * 60)

    print(
        "ANALYSIS COMPLETE"
    )

    print("=" * 60)

    print("\nCreated/updated:")

    print(
        "- outputs/research_statistics.csv"
    )

    print(
        "- outputs/age_group_statistics.csv"
    )

    print(
        "- outputs/age_group_statistical_test.csv"
    )

    print(
        "- outputs/top_k_robustness.csv"
    )

    print(
        "- outputs/top_k_robustness.png"
    )

    print(
        "- outputs/feature_frequency_comparison.csv"
    )

    print(
        "- outputs/final_research_summary.csv"
    )


if __name__ == "__main__":
    main()