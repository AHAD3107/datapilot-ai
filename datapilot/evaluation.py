"""Leakage-aware split planning and model evaluation."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class EvaluationBundle:
    task: str
    target: str
    split_strategy: str
    train_rows: int
    test_rows: int
    metrics: dict[str, dict[str, float]]
    predictions: pd.DataFrame
    warnings: list[str]
    model_status: dict[str, str]
    model_timings: dict[str, float]
    feature_columns: list[str]
    excluded_columns: list[str] = field(default_factory=list)
    total_rows: int = 0
    eligible_rows: int = 0
    dropped_target_rows: int = 0
    tabpfn_version: str | None = None


def _sanitize_error_message(msg: str, token: str | None = None) -> str:
    """Redact any potential credentials, tokens, or URL secrets."""
    sanitized = msg
    if token and len(token) >= 4:
        sanitized = sanitized.replace(token, "[REDACTED_TABPFN_TOKEN]")
    sanitized = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]{8,}", r"[REDACTED_TOKEN]", sanitized)
    sanitized = re.sub(r"(https?://[^:]+:)[^@]+(@)", r"[REDACTED_PASSWORD]", sanitized)
    sanitized = re.sub(r"((?:token|api[_-]?key)=)[^&\s]+", r"[REDACTED]", sanitized, flags=re.I)
    return sanitized


def _clean_target_mask(target_series: pd.Series) -> pd.Series:
    """Return boolean mask of valid (non-missing) target rows."""
    mask = target_series.notna()
    if pd.api.types.is_object_dtype(target_series) or pd.api.types.is_string_dtype(target_series) or isinstance(target_series.dtype, pd.CategoricalDtype):
        str_vals = target_series.astype(str).str.strip()
        mask = mask & (str_vals != "") & (~str_vals.str.lower().isin({"nan", "none", "null", "<na>"}))
    return mask


def _find_date_columns(df: pd.DataFrame) -> list[str]:
    """Identify candidate date columns in the training partition."""
    date_cols: list[str] = []
    for col in df.columns:
        name = str(col).lower()
        if pd.api.types.is_datetime64_any_dtype(df[col]) or any(
            token in name for token in ("date", "time", "created_at", "week_start", "timestamp")
        ):
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.8:
                date_cols.append(str(col))
    return date_cols


def _transform_date_columns(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    """Expand verified date columns into calendar features."""
    out = df.copy()
    for col in date_cols:
        if col in out.columns:
            parsed = pd.to_datetime(out[col], errors="coerce", format="mixed")
            out[f"{col}__year"] = parsed.dt.year
            out[f"{col}__month"] = parsed.dt.month
            out[f"{col}__day"] = parsed.dt.day
            out[f"{col}__weekday"] = parsed.dt.dayofweek
            out = out.drop(columns=[col])
    return out


def _safe_feature_frame(X: pd.DataFrame, date_cols: list[str] | None = None) -> pd.DataFrame:
    """Prepare feature DataFrame defensively without deprecated pandas operations."""
    out = X.copy()
    if date_cols:
        out = _transform_date_columns(out, date_cols)

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            mask = np.isinf(out[col])
            if mask.any():
                out.loc[mask, col] = np.nan
        else:
            if pd.api.types.is_bool_dtype(out[col]):
                out[col] = out[col].astype("object")
            if (
                pd.api.types.is_object_dtype(out[col])
                or pd.api.types.is_string_dtype(out[col])
                or isinstance(out[col].dtype, pd.CategoricalDtype)
            ):
                out[col] = out[col].astype("string").fillna("__MISSING__").astype(str)
                # Avoid transmitting long free-text blobs to the model.
                out[col] = out[col].str.slice(0, 500)
    return out


def _preprocess_for_sklearn(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical = [c for c in X_train.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=2,
                                max_categories=100,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def _make_estimator(task: str, kind: str, X_train: pd.DataFrame):
    prep = _preprocess_for_sklearn(X_train)
    if task == "classification":
        estimator = (
            DummyClassifier(strategy="prior")
            if kind == "dummy"
            else RandomForestClassifier(
                n_estimators=120,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            )
        )
    else:
        estimator = (
            DummyRegressor(strategy="median")
            if kind == "dummy"
            else RandomForestRegressor(
                n_estimators=120,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        )
    return Pipeline([("preprocess", prep), ("model", estimator)])


def _metric_values(task: str, y_true, y_pred) -> dict[str, float]:
    if task == "classification":
        try:
            acc = float(accuracy_score(y_true, y_pred))
        except Exception:
            acc = float("nan")
        try:
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        except Exception:
            bal_acc = float("nan")
        try:
            f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        except Exception:
            f1 = float("nan")
        return {
            "Accuracy": acc,
            "Balanced accuracy": bal_acc,
            "F1 (weighted)": f1,
        }

    try:
        mae = float(mean_absolute_error(y_true, y_pred))
    except Exception:
        mae = float("nan")
    try:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception:
        rmse = float("nan")
    try:
        if len(y_true) >= 2 and np.var(y_true) > 1e-12:
            r2 = float(r2_score(y_true, y_pred))
        else:
            r2 = float("nan")
    except Exception:
        r2 = float("nan")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
    }


def _split_indices(
    df: pd.DataFrame,
    time_col: str | None,
    target: str,
    task: str,
    test_size: float,
):
    valid_mask = _clean_target_mask(df[target])
    usable = df.loc[valid_mask].copy()
    total_rows = len(df)
    eligible_rows = len(usable)
    dropped_target_rows = total_rows - eligible_rows

    if eligible_rows < 30:
        raise ValueError(
            f"Only {eligible_rows} rows have a valid, non-missing target "
            f"({dropped_target_rows} missing target rows dropped out of {total_rows}). "
            "At least 30 valid rows are required for this prototype."
        )

    warnings: list[str] = []
    if dropped_target_rows > 0:
        warnings.append(
            f"Excluded {dropped_target_rows:,} row(s) with missing target values from evaluation. "
            f"Eligible evaluation population: {eligible_rows:,} rows."
        )

    if time_col and time_col in usable.columns:
        dates = pd.to_datetime(usable[time_col], errors="coerce", format="mixed")
        valid = dates.notna()
        if valid.mean() < 0.8:
            warnings.append(
                f"The selected time column '{time_col}' has too many unparseable values (less than 80% valid). "
                "Falling back to a random split."
            )
        else:
            usable_time = usable.loc[valid].copy()
            dates = dates.loc[valid]
            # Split by unique timestamps so rows from the same date never leak across partitions.
            date_order = pd.DataFrame({"_date": dates}).sort_values("_date", kind="stable")
            unique_dates = date_order["_date"].drop_duplicates().tolist()
            date_cut = int(len(unique_dates) * (1 - test_size))
            if date_cut < 2 or len(unique_dates) - date_cut < 2:
                warnings.append(
                    f"The chronological split needs at least four distinct dates (found {len(unique_dates)}). "
                    "Falling back to a random split."
                )
            else:
                train_dates = set(unique_dates[:date_cut])
                train_mask = dates.isin(train_dates)
                train_rows = usable_time.loc[train_mask]
                test_rows = usable_time.loc[~train_mask]
                if len(train_rows) < 20 or len(test_rows) < 5:
                    warnings.append(
                        f"The chronological split would leave too few rows in train ({len(train_rows)}) "
                        f"or test ({len(test_rows)}). Falling back to a random split."
                    )
                else:
                    return train_rows, test_rows, "Chronological hold-out", warnings, total_rows, eligible_rows, dropped_target_rows

    stratify = None
    if task == "classification":
        counts = usable[target].value_counts()
        if len(counts) >= 2 and counts.min() >= 2 and usable[target].nunique() <= 50:
            stratify = usable[target]
        else:
            warnings.append(
                "Could not stratify the split safely because one or more classes "
                "have fewer than two rows or there are too many classes."
            )
    train, test = train_test_split(
        usable, test_size=test_size, random_state=42, stratify=stratify
    )
    return (
        train,
        test,
        "Stratified random hold-out" if stratify is not None else "Random hold-out",
        warnings,
        total_rows,
        eligible_rows,
        dropped_target_rows,
    )


def evaluate_dataset(
    df: pd.DataFrame,
    target: str,
    task: str,
    time_col: str | None = None,
    excluded_columns: list[str] | None = None,
    run_tabpfn: bool = True,
) -> EvaluationBundle:
    if target not in df.columns:
        raise ValueError("Choose a target column that exists in the dataset.")
    if task not in {"classification", "regression"}:
        raise ValueError("Task must be classification or regression.")
    if task == "regression" and not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError("Regression requires a numeric target column.")
    if df[target].nunique(dropna=True) < 2:
        raise ValueError(
            "The target has fewer than two distinct values, so a model comparison "
            "is not meaningful."
        )

    (
        train,
        test,
        split_strategy,
        warnings,
        total_rows,
        eligible_rows,
        dropped_target_rows,
    ) = _split_indices(df, time_col, target, task, 0.2)

    raw_exclusions = list(excluded_columns or [])
    non_existent = [c for c in raw_exclusions if c not in df.columns]
    if non_existent:
        warnings.append(
            f"Specified excluded column(s) not found in dataset: {', '.join(non_existent)}. Ignored."
        )

    # Validate and exclude requested columns; ensure target is never in features
    actual_excluded = [c for c in raw_exclusions if c in df.columns and c != target]
    excluded_set = set(actual_excluded) | {target}
    X_columns = [c for c in df.columns if c not in excluded_set]

    if not X_columns:
        raise ValueError("No feature columns remain after exclusions.")

    # Determine date candidates from training partition to prevent preprocessing leakage
    date_candidates = _find_date_columns(train[X_columns])
    X_train_raw = _safe_feature_frame(train[X_columns], date_cols=date_candidates)
    X_test_raw = _safe_feature_frame(test[X_columns], date_cols=date_candidates)
    # Reindex test features to guarantee exact alignment with training features
    X_test_raw = X_test_raw.reindex(columns=X_train_raw.columns)

    y_train = train[target]
    y_test = test[target]

    if task == "classification":
        if y_train.nunique() < 2:
            raise ValueError(
                "Training partition contains only one class. Choose a better target or split."
            )
        if y_train.nunique() > 50:
            warnings.append(
                "The target has more than 50 classes; this prototype is not tuned "
                "for high-cardinality classification targets."
            )

    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, Any] = {
        "actual": (
            y_test.astype(str).to_numpy()
            if task == "classification"
            else y_test.to_numpy()
        )
    }
    timings: dict[str, float] = {}
    statuses: dict[str, str] = {}

    # Naive baseline and tree reference model always run locally.
    for kind, label in (("dummy", "Naive baseline"), ("tree", "Random forest")):
        start = time.perf_counter()
        estimator = _make_estimator(task, kind, X_train_raw)
        estimator.fit(X_train_raw, y_train)
        pred = estimator.predict(X_test_raw)
        timings[label] = time.perf_counter() - start
        metrics[label] = _metric_values(task, y_test, pred)
        predictions[label] = pred.astype(str) if task == "classification" else pred
        statuses[label] = "completed"

    tabpfn_version_used: str | None = None
    token = os.getenv("TABPFN_TOKEN", "").strip()
    if run_tabpfn and token:
        tabpfn_version_used = "v3.5"
        try:
            from tabpfn_client import TabPFNClassifier, TabPFNRegressor

            # Pin the model version so runs are easier to reproduce and document.
            if task == "classification":
                model = TabPFNClassifier.create_default_for_version(tabpfn_version_used)
            else:
                model = TabPFNRegressor.create_default_for_version(tabpfn_version_used)

            start = time.perf_counter()
            model.fit(X_train_raw, y_train)
            pred = model.predict(X_test_raw)
            timings["TabPFN"] = time.perf_counter() - start
            metrics["TabPFN"] = _metric_values(task, y_test, pred)
            predictions["TabPFN"] = (
                pred.astype(str) if task == "classification" else pred
            )
            statuses["TabPFN"] = "completed"
        except Exception as exc:
            statuses["TabPFN"] = "failed"
            raw_msg = str(exc)
            clean_msg = _sanitize_error_message(raw_msg, token)[:240]
            exc_type = type(exc).__name__

            if any(t in raw_msg.lower() for t in ("unauthorized", "authentication", "invalid token", "forbidden", "401", "403")):
                reason_desc = "Authentication failed (invalid or expired token)"
            elif any(t in raw_msg.lower() for t in ("timeout", "timed out", "timeouterror")):
                reason_desc = "Service request timed out"
            elif any(t in raw_msg.lower() for t in ("connect", "connection error", "502", "503", "504")):
                reason_desc = "Connection or server availability error"
            else:
                reason_desc = f"{exc_type}: {clean_msg}"

            warnings.append(
                f"TabPFN inference failed ({reason_desc}). "
                "Local baseline results are still available; no TabPFN score was fabricated."
            )
    elif run_tabpfn:
        statuses["TabPFN"] = "not configured"
        warnings.append(
            "TABPFN_TOKEN is not configured. Only local baseline metrics were calculated. "
            "Configure the hosted TabPFN API token to complete the partner-model comparison. "
            "No TabPFN result was fabricated."
        )
    else:
        statuses["TabPFN"] = "skipped"

    pred_df = pd.DataFrame(predictions)
    pred_df.insert(0, "source_row", test.index.astype(str))
    return EvaluationBundle(
        task=task,
        target=target,
        split_strategy=split_strategy,
        train_rows=len(train),
        test_rows=len(test),
        metrics=metrics,
        predictions=pred_df,
        warnings=warnings,
        model_status=statuses,
        model_timings=timings,
        feature_columns=list(X_train_raw.columns),
        excluded_columns=actual_excluded,
        total_rows=total_rows,
        eligible_rows=eligible_rows,
        dropped_target_rows=dropped_target_rows,
        tabpfn_version=tabpfn_version_used,
    )


def build_decision(bundle: EvaluationBundle) -> dict[str, str]:
    """Produce conservative, explainable labels against the strongest local baseline."""
    # Check for unexcluded target leakage in the active feature matrix
    leakage_warning = any(
        ("Critical target leakage detected" in w or "Potential target leakage" in w or "Suspected target proxy detected" in w)
        for w in bundle.warnings
    )

    model = bundle.metrics.get("TabPFN")
    candidates = {
        name: metrics
        for name, metrics in bundle.metrics.items()
        if name in {"Naive baseline", "Random forest"}
    }
    if not model:
        base_decision = {
            "label": "INCOMPLETE EVALUATION",
            "reason": (
                "TabPFN was not evaluated successfully. Local baselines are available, "
                "but the partner-model comparison is incomplete."
            ),
        }
        if leakage_warning:
            base_decision["reason"] += " Note: Potential target leakage was detected in the dataset; verify exclusions before using baseline models."
        return base_decision

    metric_name = "MAE" if bundle.task == "regression" else "Balanced accuracy"
    candidates = {
        name: metrics for name, metrics in candidates.items() if metric_name in metrics
    }
    if not candidates or metric_name not in model:
        return {
            "label": "INSUFFICIENT EVIDENCE",
            "reason": "The expected metrics are unavailable for a fair comparison.",
        }
    if bundle.task == "regression":
        best_name, best_metrics = min(
            candidates.items(), key=lambda item: item[1][metric_name]
        )
        base = best_metrics[metric_name]
        value = model[metric_name]
        if base <= 0:
            return {
                "label": "NEEDS INVESTIGATION",
                "reason": (
                    f"{best_name} has zero or near-zero MAE, so relative improvement "
                    "is not meaningful."
                ),
            }
        improvement = (base - value) / base
        if bundle.test_rows < 30:
            return {
                "label": "INSUFFICIENT EVIDENCE",
                "reason": (
                    "Fewer than 30 held-out rows make the estimate too unstable "
                    "for a confident recommendation."
                ),
            }
        if improvement >= 0.10:
            label, reason = (
                "PROMISING",
                f"TabPFN reduced held-out MAE by {improvement:.1%} versus the strongest "
                f"local baseline ({best_name}). This is encouraging but does not establish "
                "future performance.",
            )
        elif improvement > 0:
            label, reason = (
                "NEEDS INVESTIGATION",
                f"TabPFN improved held-out MAE by only {improvement:.1%} versus {best_name}. "
                "Check feature quality and temporal drift before relying on it.",
            )
        else:
            label, reason = (
                "NEEDS INVESTIGATION",
                f"TabPFN did not beat the strongest local baseline ({best_name}) on held-out "
                f"MAE (change: {improvement:.1%}). Prefer the simpler or lower-error model "
                "until further testing explains the result.",
            )
    else:
        best_name, best_metrics = max(
            candidates.items(), key=lambda item: item[1][metric_name]
        )
        base = best_metrics[metric_name]
        value = model[metric_name]
        improvement = value - base
        if bundle.test_rows < 30:
            return {
                "label": "INSUFFICIENT EVIDENCE",
                "reason": (
                    "Fewer than 30 held-out rows make the estimate too unstable "
                    "for a confident recommendation."
                ),
            }
        if improvement >= 0.05:
            label, reason = (
                "PROMISING",
                f"TabPFN improved held-out balanced accuracy by {improvement:.3f} versus "
                f"the strongest local baseline ({best_name}). Review per-class results before use.",
            )
        elif improvement > 0:
            label, reason = (
                "NEEDS INVESTIGATION",
                f"TabPFN improved balanced accuracy by only {improvement:.3f} versus {best_name}. "
                "Check class imbalance and use a second validation split before relying on it.",
            )
        else:
            label, reason = (
                "NEEDS INVESTIGATION",
                f"TabPFN did not beat the strongest local baseline ({best_name}) on held-out "
                f"balanced accuracy (change: {improvement:.3f}). Avoid relying on it without "
                "further validation.",
            )

    if leakage_warning:
        label = "NEEDS INVESTIGATION"
        reason += " Potential target leakage or proxy signals were detected; investigate and exclude suspicious features before relying on any model metrics."
    return {"label": label, "reason": reason}
