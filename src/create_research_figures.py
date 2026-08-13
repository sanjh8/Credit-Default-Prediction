import os

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)


def main():

    print("=" * 60)
    print("CREATING RESEARCH FIGURES")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. SHAP vs LIME stability
    # --------------------------------------------------------

    stability_file = os.path.join(
        OUTPUT_DIR,
        "stability_summary.csv"
    )

    stability_test_file = os.path.join(
        OUTPUT_DIR,
        "stability_statistical_test.csv"
    )

    stability = pd.read_csv(
        stability_file
    )

    stability_test = pd.read_csv(
        stability_test_file
    )

    x = stability["top_k"]

    plt.figure(figsize=(8, 5))

    plt.plot(
        x,
        stability["mean_shap_stability"],
        marker="o",
        linewidth=2,
        label="SHAP"
    )

    plt.plot(
        x,
        stability["mean_lime_stability"],
        marker="o",
        linewidth=2,
        label="LIME"
    )

    plt.xlabel("Number of Top Features (K)")
    plt.ylabel("Explanation Stability")
    plt.title(
        "SHAP vs LIME Explanation Stability"
    )

    plt.xticks([3, 5, 10])
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    stability_plot = os.path.join(
        OUTPUT_DIR,
        "shap_lime_stability_comparison.png"
    )

    plt.savefig(
        stability_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # 2. SHAP-LIME agreement across Top-K
    # --------------------------------------------------------

    robustness_file = os.path.join(
        OUTPUT_DIR,
        "top_k_robustness.csv"
    )

    robustness = pd.read_csv(
        robustness_file
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        robustness["top_k"],
        robustness["mean_jaccard"],
        marker="o",
        linewidth=2
    )

    plt.xlabel("Number of Top Features (K)")
    plt.ylabel("Mean Jaccard Agreement")
    plt.title(
        "SHAP-LIME Agreement Across Top-K Features"
    )

    plt.xticks([3, 5, 10])
    plt.ylim(0, 0.55)
    plt.grid(alpha=0.3)

    plt.tight_layout()

    agreement_plot = os.path.join(
        OUTPUT_DIR,
        "shap_lime_agreement_top_k.png"
    )

    plt.savefig(
        agreement_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # 3. Age-group agreement
    # --------------------------------------------------------

    age_file = os.path.join(
        OUTPUT_DIR,
        "age_group_statistics.csv"
    )

    age = pd.read_csv(
        age_file
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        age["age_group"],
        age["mean"]
    )

    plt.xlabel("Age Group")
    plt.ylabel("Mean Jaccard Agreement")
    plt.title(
        "SHAP-LIME Agreement Across Age Groups"
    )

    plt.ylim(0, 0.55)
    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    age_plot = os.path.join(
        OUTPUT_DIR,
        "shap_lime_agreement_age_groups.png"
    )

    plt.savefig(
        age_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\nFigures created:")

    print(
        f"- {stability_plot}"
    )

    print(
        f"- {agreement_plot}"
    )

    print(
        f"- {age_plot}"
    )

    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()