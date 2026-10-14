# DEV submission draft — DataPilot AI

> Replace the live URL, repository URL, screenshots and all results below with verified values before publishing. Do not claim an API run that you did not perform.

## Title

**DataPilot AI: A Prediction-Readiness Auditor That Tells You When Not to Trust Your Model**

## The problem

Small businesses often have plenty of spreadsheet data but lack the expertise to judge whether that data supports a useful prediction. Missing values, suspicious features, leakage, and weak validation can make a model look convincing while giving the business owner little reason to trust it.

## What I built

DataPilot AI profiles a CSV, flags potential data-quality and target-leakage signals, uses a hold-out split chosen for the dataset, compares local baseline models with TabPFN, and produces a decision report based on measured performance. It is designed for freelancers, online sellers, and small-business owners who want to test a prediction before using it in a decision.

## Demo

[Add deployed application URL or a video demonstration]

## Source code

[Add public GitHub repository URL]

## Why open innovation matters

DataPilot uses inspectable Python checks and reproducible evaluation around TabPFN's tabular foundation model. It separates computed metrics from interpretation, gives users direct access to validation choices and warnings, and allows the evaluation pipeline to evolve without depending on a single closed AI assistant. The app documents that hosted TabPFN inference transmits relevant dataset values to Prior Labs, so users can make an informed decision about what data they send.

## Technical execution and actual results

[Replace with actual tested values: dataset shape, split strategy, local baseline metrics, TabPFN metrics and whether it improved over the strongest baseline. Include one example warning and an honest failure case.]

## Limitations

This is a prototype. Heuristic leakage checks do not prove a dataset is safe or unsafe. A single hold-out metric does not guarantee future performance. Group-aware validation, nested cross-validation, drift monitoring and deeper feature-availability checks are future work. The model's current licence and API terms must be reviewed before commercial use.

## Prize categories

- Best Use of TabPFN (only if the submitted demo successfully integrates TabPFN)

#hf26challenge
