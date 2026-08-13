# Credit Default Prediction with Explainable AI

A machine learning project for predicting credit card default and analyzing the consistency of local explanations produced by SHAP and LIME.

The project uses the UCI **Default of Credit Card Clients** dataset and evaluates Logistic Regression, Random Forest, and XGBoost models. The main analysis focuses on the agreement between SHAP and LIME explanations for individual predictions.

## 1. Project Overview

Credit-default prediction models can achieve strong predictive performance while remaining difficult to interpret.

This project investigates two questions:

1. How consistently do SHAP and LIME identify important features for the same credit-default prediction?
2. Does explanation agreement vary across customer age groups or according to the number of top features considered?

The explanation agreement is measured using **Jaccard similarity** between the features identified as important by SHAP and LIME.

## 2. Dataset

This project uses the UCI **Default of Credit Card Clients** dataset.

The dataset contains information on 30,000 credit card customers in Taiwan and includes demographic information, credit limits, payment history, bill amounts, payment amounts, and whether the customer defaulted on their payment in the following month.

Dataset source:

https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

### Dataset setup

1. Download the dataset from UCI.
2. Extract `default of credit card clients.xls`.
3. Rename it to:

```text
credit_card_default.xls
```

4. Place it inside:

```text
data/credit_card_default.xls
```

The `.xls` file requires the `xlrd` package, which is included in `requirements.txt`.

## 3. Project Structure

```text
Credit-Default-Prediction/
│
├── data/
│   ├── credit_card_default.xls
│   ├── train.csv
│   └── test.csv
│
├── outputs/
│   ├── model_comparison.csv
│   ├── shap_lime_agreement.csv
│   ├── agreement_by_age_group.csv
│   ├── agreement_histogram.png
│   ├── shap_summary.png
│   ├── age_group_statistics.csv
│   ├── age_group_statistical_test.csv
│   ├── top_k_robustness.csv
│   ├── top_k_robustness.png
│   ├── feature_frequency_comparison.csv
│   └── final_research_summary.csv
│
├── src/
│   ├── preprocess.py
│   ├── train_models.py
│   ├── explain.py
│   └── research_analysis.py
│
├── README.md
└── requirements.txt
```

## 4. Installation

Create and activate a virtual environment.

### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

## 5. Run the Pipeline

### Step 1 — Preprocess the dataset

```powershell
python src/preprocess.py
```

This removes the ID column, normalizes undocumented categorical codes, performs a stratified train/test split, and creates:

```text
data/train.csv
data/test.csv
```

### Step 2 — Train the models

```powershell
python src/train_models.py
```

Three models are evaluated:

- Logistic Regression
- Random Forest
- XGBoost

The evaluation includes:

- AUC-ROC
- Average Precision (PR-AUC)
- F1-score
- Precision
- Recall

Model results are saved to:

```text
outputs/model_comparison.csv
```

### Step 3 — Generate SHAP and LIME explanations

```powershell
python src/explain.py
```

The script generates local explanations for 200 age-stratified test samples and calculates SHAP-LIME agreement using Jaccard similarity for:

- Top-3 features
- Top-5 features
- Top-10 features

The results are saved to:

```text
outputs/shap_lime_agreement.csv
```

### Step 4 — Run statistical analysis

```powershell
python src/research_analysis.py
```

This performs:

- Bootstrap confidence intervals
- Age-group comparison
- Kruskal-Wallis statistical testing
- Epsilon-squared effect size calculation
- Top-K robustness analysis
- Feature-level frequency analysis

The resulting statistics are saved in the `outputs/` directory.

## 6. Current Experimental Results

Using the current 200-sample experiment, the mean Top-5 SHAP-LIME Jaccard agreement is approximately:

```text
0.4157
```

with a 95% bootstrap confidence interval of approximately:

```text
0.3921 – 0.4390
```

The observed mean agreement increases as more top features are considered:

| Top-K | Mean Jaccard Agreement |
|------:|-----------------------:|
| 3     | 0.3755 |
| 5     | 0.4157 |
| 10    | 0.4344 |

Age-group analysis did not detect a statistically significant difference in explanation agreement:

```text
Kruskal-Wallis H = 0.7658
p = 0.8576
```

These results are preliminary experimental findings and should not be interpreted as evidence that SHAP or LIME is universally more reliable.

## 7. Research Focus

The research focuses on **explanation consistency** rather than simply comparing predictive performance.

The central idea is to investigate whether two widely used explainability methods produce similar feature-level explanations for the same underlying machine learning model.

The analysis considers:

- Cross-method explanation agreement
- Agreement across different explanation granularities
- Agreement across customer age groups
- Feature-level differences between SHAP and LIME

Further experiments will evaluate explanation stability and strengthen the statistical analysis.

## 8. Limitations

The current study has several limitations:

- The analysis uses a single public credit-default dataset.
- The SHAP-LIME comparison currently focuses on XGBoost explanations.
- The explanation sample contains 200 test customers.
- Jaccard similarity measures feature-set overlap but does not directly compare attribution magnitude or direction.
- Age is currently the primary subgroup variable investigated.
- Additional datasets and stability experiments would be required to support broader conclusions.

## 9. Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- LIME
- Matplotlib
- SciPy
- Joblib

## 10. License

This project is intended for academic and research purposes.