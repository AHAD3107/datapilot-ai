# DataPilot AI

**Before you trust a prediction, test the evidence.**

DataPilot AI is an open-source prototype that audits a tabular dataset before a user relies on a prediction. It profiles data quality, flags heuristic leakage signals, evaluates baseline models on held-out records, and compares them with hosted TabPFN when configured. It then exports an evidence-backed prediction-readiness report.

> **Status:** Hackathon MVP / research prototype. It does not guarantee future model performance or certify that data is free from leakage.

## Features

- CSV upload with a 10 MB limit and predictable row/column caps.
- Dataset overview: missingness, duplicates, unique values, constant columns and cardinality warnings.
- Target selection for classification or regression.
- Random/stratified holdout or chronological split by date; rows sharing a date stay in the same partition.
- Naive baseline and Random Forest reference model run locally.
- Optional TabPFN via the official `tabpfn-client` hosted API.
- Compare model metrics on the same held-out rows.
- Conservative decision labels: Promising, Needs investigation, or Incomplete evaluation.
- Download held-out predictions, metrics and Markdown audit report.
- Deterministic synthetic retail sales sample data and tests.

## Screens / demo

Add real screenshots of the running application under `docs/screenshots/` before publication. Do not use synthetic screenshots as evidence of a completed hosted run.

## Quick start (Windows PowerShell)

Recommended: Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app opens in the browser at the local Streamlit URL printed in the terminal. You can use the built-in synthetic sales dataset without an API key; local baseline models run, while the TabPFN panel explicitly shows that the partner-model comparison is not configured.

## Enable TabPFN

1. Create an API token at [Prior Labs](https://platform.priorlabs.ai/account/api-keys).
2. Set it in your local PowerShell session:

   ```powershell
   $env:TABPFN_TOKEN = "YOUR_TOKEN_HERE"
   ```

3. Restart the app and run the audit again.

**Data handling:** the official TabPFN Client uses hosted inference and transmits relevant dataset values to Prior Labs' servers. Use only data you have permission to share with the service. Do not commit API tokens. Review the provider's current model licence, terms, quotas and prices before use—especially before commercial deployment.

## Cloud deployment: Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
3. Select the repository, branch `main`, and entry point `app.py`.
4. In advanced settings, choose a supported Python version (3.12 is the tested target for this project).
5. If you have a TabPFN token, add it to the app's Secrets field:

   ```toml
   TABPFN_TOKEN = "YOUR_TOKEN_HERE"
   ```

6. Deploy, then open the public URL and test the sample dataset from a fresh browser session.

The app works without the token, but its TabPFN comparison remains incomplete. Do not claim TabPFN results unless the API call actually succeeded. See Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [secrets documentation](https://docs.streamlit.io/deploy/concepts/secrets).

## Run tests

```powershell
python -m pytest -q
```

Tests use local baseline models and do not call hosted TabPFN, so they do not need API credits.

## How evaluation works

- Regression: MAE, RMSE and R²; lower MAE/RMSE is better.
- Classification: accuracy, balanced accuracy and weighted F1; higher is better.
- The same held-out rows are used for all models.
- Preprocessing pipelines for local baseline models are fit using training records only.
- When a usable time column is selected, all rows on a timestamp are kept in the same chronological partition.
- Potential leakage is a heuristic warning for human review, not proof.

## Repository map

```text
app.py                       Streamlit interface
 datapilot/profiling.py      Data profile and warnings
 datapilot/evaluation.py     Splits, baselines, TabPFN and metrics
 datapilot/report.py         Markdown report generator
 datapilot/sample_data.py    Deterministic synthetic dataset
 tests/                       Automated tests
data/sample_sales.csv         Sample CSV
```

## Open innovation

DataPilot uses inspectable data checks and reproducible Python evaluation around an open software interface to TabPFN. It separates measured metrics from generated explanation, lets users inspect the validation strategy, and documents the hosted-inference dependency and limitations.

## Licensing

The project source is licensed under MIT. This licence applies to this repository's original code, not to third-party packages, TabPFN model assets, or the hosted service. Review each dependency and provider's terms separately.
