# Credit Risk Prediction with Explainable AI (SHAP vs LIME)

A project comparing model performance and explanation consistency (SHAP vs LIME)
for credit default prediction, using the UCI "Default of Credit Card Clients" dataset.

## 1. Setup

```bash
cd credit-risk-xai
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the dataset

We're using the **UCI "Default of Credit Card Clients" dataset** — 30,000 credit
card customers in Taiwan, with payment history and whether they defaulted the
next month. No login or account needed.

1. Go to https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
2. Click **"Download"** — you'll get a zip file, no sign-in required.
3. Unzip it — inside is `default of credit card clients.xls`
4. Rename it to `credit_card_default.xls` and place it in the `data/` folder, so
   you have: `credit-risk-xai/data/credit_card_default.xls`

(Reading `.xls` needs the `xlrd` package — it's already in `requirements.txt`.)

## 3. Run the pipeline

```bash
python src/preprocess.py       # cleans data, saves data/processed.csv
python src/train_models.py     # trains LogReg, RandomForest, XGBoost; saves models + metrics
python src/explain.py          # runs SHAP + LIME, compares them, saves plots to outputs/
```

## 4. What each script does

- **preprocess.py** — loads the UCI dataset, removes the ID column, normalizes
  undocumented EDUCATION and MARRIAGE codes, performs a stratified train/test
  split, and saves train.csv and test.csv.
- **train_models.py** — trains 3 models (Logistic Regression as interpretable
  baseline, Random Forest, XGBoost), evaluates with AUC-ROC / F1 / precision-recall
  (important since defaults are a minority class), saves the best model.
- **explain.py** — generates SHAP values and LIME explanations for the same set
  of test customers, then computes an agreement score between the two methods
  (this comparison is your paper's novelty angle).

## 5. Next steps for the paper

- Run `explain.py` on 200 age-stratified test samples and look at the `shap_lime_agreement.csv`
  output — this is your core result table.
- Try splitting customers into subgroups (e.g., by age or income bracket) and see
  if SHAP/LIME agreement differs across groups — this is a stronger, more specific
  research question than "SHAP vs LIME" alone.
- Optional: add a fairness check (do explanations reveal reliance on proxy variables
  like age?).
