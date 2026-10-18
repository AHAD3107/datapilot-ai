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



## AI Analyst & Grounded Audit Explanations (Phase A)

DataPilot AI includes a grounded **AI Analyst** capable of explaining data quality findings, leakage risks, validation limitations, and baseline model comparisons in plain English.

### Grounding & Zero Raw-Data Transmission
- **Strict Data Sanitization:** The AI Analyst never receives raw CSV rows, feature values, user credentials, or personal information.
- **Bounded Audit Context:** Only high-level aggregated audit metadata (dataset dimensions, row counts, evaluated metric scores, detected leakage flags, corrupted-numeric column names, and warning messages) is serialized into an untrusted metadata block for the LLM.
- **Adversarial Hardening:** Untrusted column names and dataset metadata are sanitized and strictly quarantined from system instructions to prevent prompt-injection attacks.
- **Fact vs. Recommendation:** The analyst is strictly instructed to distinguish facts from recommendations, honestly state when information is unavailable, and never fabricate metrics or declare leakage definitively without proof.

### Deterministic vs. LLM-Generated Explanations
DataPilot AI provides two explanation paths:
1. **Google Gemini LLM Explanations:** When configured and external inference is permitted, Gemini provides interactive Q&A and narrative executive summaries grounded strictly in the `AuditContext`.
2. **Deterministic Local Fallback:** When the LLM is unavailable, unconfigured, rate-limited, or blocked by Privacy Mode, DataPilot generates deterministic rule-based explanations and executive summaries directly from Python. No fabricated scores or simulated AI responses are produced.

### Privacy Mode (Application Air-Gap Control)
A prominent **Disable external AI calls (Privacy Mode)** toggle is available in the application sidebar and can also be enforced via `DATAPILOT_AIRGAP=true`.
- **What it blocks:** When enabled, the application's central policy engine interceptor strictly blocks all outbound network calls to external AI services:
  - Google Gemini API calls are blocked.
  - Hosted TabPFN inference API calls are blocked.
- **What remains functional:** Local profiling, data leakage detection, train/test splits, local baseline models (Naive baseline and Random Forest), deterministic executive summaries, and full Markdown report exports run 100% locally.
- **Scope limitation:** Privacy Mode is an explicit application-level integration gate. It prevents DataPilot's Python code from calling external AI APIs; it is not an operating-system firewall.

### Configuring Google Gemini
1. Obtain an API key from [Google AI Studio](https://aistudio.google.com/).
2. Set the environment variables in your terminal or `.env`:
   ```powershell
   $env:AI_PROVIDER = "gemini"
   $env:AI_API_KEY = "YOUR_GEMINI_API_KEY"
   $env:AI_MODEL = "gemini-2.5-flash"
   ```
3. The model defaults to `gemini-2.5-flash`. You can configure any supported Gemini model ID via `AI_MODEL`.

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
app.py                       Streamlit interface (with Privacy Mode & AI Analyst)
 datapilot/analyst.py        Grounded AI Analyst adapter (Gemini & deterministic)
 datapilot/context.py        Sanitized AuditContext DTO & prompt serialization
 datapilot/policy.py         Outbound inference privacy policy engine
 datapilot/profiling.py      Data profile, leakage heuristics & corrupted-numeric detection
 datapilot/evaluation.py     Splits, baselines, TabPFN and metrics
 datapilot/report.py         Executive summary & Markdown report generator
 datapilot/sample_data.py    Deterministic synthetic dataset
 tests/                      Automated test suite
data/sample_sales.csv        Sample CSV
```

## Open innovation

DataPilot uses inspectable data checks and reproducible Python evaluation around an open software interface to TabPFN. It separates measured metrics from generated explanation, lets users inspect the validation strategy, and documents the hosted-inference dependency and limitations.

## Licensing

The project source is licensed under MIT. This licence applies to this repository's original code, not to third-party packages, TabPFN model assets, or the hosted service. Review each dependency and provider's terms separately.
