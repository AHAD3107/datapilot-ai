# DataPilot AI architecture

## Main flow

1. Load a user-provided CSV (or deterministic synthetic sales data).
2. Profile rows, columns, missing values, duplicates, constant values, likely identifiers, cardinality, and heuristic leakage indicators.
3. Select a target and task; optionally split chronologically by a date field.
4. Train a naive baseline and random-forest reference model on the training partition only.
5. If `TABPFN_TOKEN` is configured, send training features/target to Prior Labs' hosted TabPFN API and evaluate predictions on the same held-out partition.
6. Generate a report from computed metrics and surfaced caveats.

## Trust boundaries

- The CSV is untrusted input. It is size-limited in the UI and row/column counts are capped.
- Uploaded data remains in the app process except when TabPFN is enabled. TabPFN Client is a cloud-based service; its API receives data for model processing.
- Never set a TabPFN token in source code. Use environment variables locally and platform secrets in cloud deployments.
- The report uses deterministic metric code. It does not rely on an LLM to invent or calculate performance scores.

## Evaluation design

- Random or stratified holdout by default; chronological holdout when a parseable time column is selected.
- In chronological splitting, all rows on the same timestamp are kept in one partition.
- Numeric imputers and category encoders for sklearn baselines are fit using training data only.
- TabPFN and baseline metrics use the same held-out rows.
- Target-leakage checks are heuristics. They cannot prove that features leak target information or that a dataset is free from leakage.

## Known limitations

- No group-aware split or nested cross-validation yet.
- No automated causal analysis or guarantee of future accuracy.
- The date feature engineering is basic calendar decomposition only.
- The hosted TabPFN API needs an account/token and may have quotas or usage costs.
- This is an MVP, not a production data governance system.
