"""Leakage-aware split planning and model evaluation."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
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


def _expand_date_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Expand recognizable date columns into numeric calendar features."""
    out = X.copy()
    candidates = []
    for col in out.columns:
        name = str(col).lower()
        if pd.api.types.is_datetime64_any_dtype(out[col]) or any(
            token in name
            for token in ("date", "time", "created_at", "week_start", "timestamp")
        ):
            parsed = pd.to_datetime(out[col], errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.8:
                candidates.append((col, parsed))
    for col, parsed in candidates:
        out[f"{col}__year"] = parsed.dt.year
        out[f"{col}__month"] = parsed.dt.month
        out[f"{col}__day"] = parsed.dt.day
        out[f"{col}__weekday"] = parsed.dt.dayofweek
        out = out.drop(columns=[col])
    return out


def _safe_feature_frame(X: pd.DataFrame) -> pd.DataFrame:
    X = _expand_date_columns(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype("object")
        if (
            pd.api.types.is_object_dtype(X[col])
            or pd.api.types.is_string_dtype(X[col])
            or isinstance(X[col].dtype, pd.CategoricalDtype)
        ):
            X[col] = X[col].astype("string").fillna("__MISSING__").astype(str)
            # Avoid transmitting long free-text blobs to the hosted tabular model.
            X[col] = X[col].str.slice(0, 500)
    return X


def _preprocess_for_sklearn(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric = [
        c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])
    ]
    categorical = [c for c in X_train.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median", keep_empty_features=True
                            ),
                        )
                    ]
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
                        ("imputer", SimpleImputer(strategy="most_frequent")),
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
    return ColumnTransformer(
        transformers=transformers, remainder="drop", sparse_threshold=0.3
    )


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
        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Balanced accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "F1 (weighted)": float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
        }
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def _split_indices(
    df: pd.DataFrame,
    time_col: str | None,
    target: str,
    task: str,
    test_size: float,
):
    usable = df.loc[df[target].notna()].copy()
    if len(usable) < 30:
        raise ValueError(
            f"Only {len(usable)} rows have a non-missing target. "
            "At least 30 rows are required for this prototype."
        )
    warnings: list[str] = []
    if time_col and time_col in usable.columns:
        dates = pd.to_datetime(usable[time_col], errors="coerce", format="mixed")
        valid = dates.notna()
        if valid.mean() < 0.8:
            warnings.append(
                f"The selected time column '{time_col}' has too many "
                "unparseable values. Used a random split instead."
            )
        else:
            usable = usable.loc[valid].copy()
            dates = dates.loc[valid]
            # Split by unique timestamps so rows from the same date never leak across partitions.
            date_order = pd.DataFrame({"_date": dates}).sort_values(
                "_date", kind="stable"
            )
            unique_dates = date_order["_date"].drop_duplicates().tolist()
            date_cut = int(len(unique_dates) * (1 - test_size))
            if date_cut < 2 or len(unique_dates) - date_cut < 2:
                raise ValueError(
                    "The chronological split needs at least four distinct dates. "
                    "Use a larger dataset or a different split."
                )
            train_dates = set(unique_dates[:date_cut])
            train_mask = dates.isin(train_dates)
            train_rows = usable.loc[train_mask]
            test_rows = usable.loc[~train_mask]
            if len(train_rows) < 20 or len(test_rows) < 5:
                raise ValueError(
                    "The chronological split would leave too few rows in train "
                    "or test. Use a larger dataset or a different split."
                )
            return train_rows, test_rows, "Chronological hold-out", warnings
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

    train, test, split_strategy, warnings = _split_indices(
        df, time_col, target, task, 0.2
    )
    excluded = set(excluded_columns or []) | {target}
    X_columns = [c for c in df.columns if c not in excluded]
    if time_col and time_col in X_columns:
        # Keep calendar components only, never the raw date string.
        pass
    if not X_columns:
        raise ValueError("No feature columns remain after exclusions.")
    X_train_raw = _safe_feature_frame(train[X_columns])
    X_test_raw = _safe_feature_frame(test[X_columns])
    # Align columns in case a conversion or source schema produces different output.
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
        "actual": y_test.astype(str).to_numpy()
        if task == "classification"
        else y_test.to_numpy()
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

    token = os.getenv("TABPFN_TOKEN", "").strip()
    if run_tabpfn and token:
        try:
            from tabpfn_client import TabPFNClassifier, TabPFNRegressor

            # Pin the model version so runs are easier to reproduce and document.
            if task == "classification":
                model = TabPFNClassifier.create_default_for_version("v3.5")
            else:
                model = TabPFNRegressor.create_default_for_version("v3.5")

            start = time.perf_counter()
            model.fit(X_train_raw, y_train)
            pred = model.predict(X_test_raw)
            timings["TabPFN"] = time.perf_counter() - start
            metrics["TabPFN"] = _metric_values(task, y_test, pred)
            predictions["TabPFN"] = pred.astype(str) if task == "classification" else pred
            statuses["TabPFN"] = "completed"
        except Exception as exc:  # Report upstream/API failure; never invent a score.
            statuses["TabPFN"] = "failed"
            warnings.append(
                f"TabPFN inference failed: {type(exc).__name__}: {str(exc)[:240]}. "
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
    )


def build_decision(bundle: EvaluationBundle) -> dict[str, str]:
    """Produce conservative, explainable labels against the strongest local baseline."""
    model = bundle.metrics.get("TabPFN")
    candidates = {
        name: metrics
        for name, metrics in bundle.metrics.items()
        if name in {"Naive baseline", "Random forest"}
    }
    if not model:
        return {
            "label": "INCOMPLETE EVALUATION",
            "reason": (
                "TabPFN was not evaluated successfully. Local baselines are available, "
                "but the partner-model comparison is incomplete."
            ),
        }

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

    if any("Potential target leakage" in warning for warning in bundle.warnings):
        if label == "PROMISING":
            label = "NEEDS INVESTIGATION"
            reason += " A potential leakage signal was flagged; investigate it before using the result."
    return {"label": label, "reason": reason}
