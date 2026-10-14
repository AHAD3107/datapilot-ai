"""Dataset profiling and transparent heuristic warnings for DataPilot AI."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def _is_date_like(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    key = name.lower()
    if not any(token in key for token in ("date", "time", "created", "updated", "week", "month", "period")):
        return False
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str).head(100)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) >= 0.8


def infer_task(target: pd.Series) -> str:
    """Suggest a task type; the user may override it in the UI."""
    y = target.dropna()
    if y.empty:
        return "classification"
    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_object_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"
    unique = int(y.nunique())
    if unique <= 12:
        return "classification"
    return "regression"


def profile_dataframe(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    if df.empty:
        raise ValueError("The dataset has no rows.")

    rows, columns = df.shape
    missing = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())
    constant_columns = [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1]
    identifier_like: list[str] = []
    high_cardinality: list[str] = []
    date_columns: list[str] = []
    for col in df.columns:
        series = df[col]
        name = str(col).lower()
        if _is_date_like(series, str(col)):
            date_columns.append(str(col))
        if re.search(r"(^|[_\W])(id|uuid|email|phone|customer_id|user_id)([_\W]|$)", name) and series.nunique(dropna=True) / max(rows, 1) > 0.5:
            identifier_like.append(str(col))
        if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and series.nunique(dropna=True) > min(50, max(20, int(rows * 0.5))):
            high_cardinality.append(str(col))

    warnings: list[dict[str, str]] = []
    if duplicate_rows:
        warnings.append({"level": "warning", "title": "Duplicate rows found", "detail": f"{duplicate_rows:,} exact duplicate rows ({duplicate_rows / max(rows, 1):.1%}). Confirm whether these are repeated events or accidental duplicates."})
    missing_cols = {str(c): int(v) for c, v in missing.items() if v > 0}
    if missing_cols:
        top = sorted(missing_cols.items(), key=lambda item: item[1], reverse=True)[:4]
        preview = ", ".join(f"{name} ({count:,})" for name, count in top)
        warnings.append({"level": "warning", "title": "Missing values detected", "detail": f"Missing values occur in {len(missing_cols)} column(s). Top columns: {preview}. Imputation must be learned on the training split only."})
    if constant_columns:
        warnings.append({"level": "info", "title": "Constant columns", "detail": "These columns contain one distinct value and probably add no predictive signal: " + ", ".join(constant_columns[:8]) + ("…" if len(constant_columns) > 8 else ".")})
    if identifier_like:
        warnings.append({"level": "warning", "title": "Possible identifier leakage", "detail": "High-cardinality identifier-like columns: " + ", ".join(identifier_like[:8]) + ". IDs can encourage memorization or reveal row identity; review before modeling."})
    if high_cardinality:
        warnings.append({"level": "warning", "title": "High-cardinality text columns", "detail": "Potentially expensive or unhelpful columns: " + ", ".join(high_cardinality[:8]) + ". Consider dropping, grouping, or deliberately engineering these features."})
    if rows < 100:
        warnings.append({"level": "warning", "title": "Small dataset", "detail": f"Only {rows} rows are available. A single hold-out result may be unstable; treat metrics as exploratory."})
    if rows < 30:
        warnings.append({"level": "error", "title": "Very small dataset", "detail": "Fewer than 30 rows are available. DataPilot will block model evaluation by default because a hold-out comparison is unlikely to be meaningful."})

    leakage_terms = re.compile(r"(^|[_\W])(future|next|post|after|outcome|result|target|label|actual|final|returned|refund_issued|resolved)([_\W]|$)", re.I)
    suspicious = []
    if target:
        target_words = [part for part in re.split(r"[_\W]+", target.lower()) if len(part) > 3]
        for col in df.columns:
            if col == target:
                continue
            name = str(col)
            if leakage_terms.search(name) or (target_words and all(word in name.lower() for word in target_words) and any(t in name.lower() for t in ("actual", "final", "after", "result", "future"))):
                suspicious.append(name)
        if suspicious:
            warnings.append({"level": "warning", "title": "Potential target leakage", "detail": "Review these columns carefully: " + ", ".join(suspicious[:10]) + ". This is a name-based heuristic, not proof of leakage. Remove features that would not exist at prediction time."})

    if target and target in df.columns:
        y = df[target].dropna()
        target_summary = {
            "name": target,
            "missing": int(df[target].isna().sum()),
            "unique": int(y.nunique()),
            "inferred_task": infer_task(df[target]),
            "dtype": str(df[target].dtype),
        }
    else:
        target_summary = None

    column_rows = []
    for col in df.columns:
        column_rows.append({
            "column": str(col),
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isna().sum()),
            "missing_pct": round(float(df[col].isna().mean() * 100), 2),
            "unique": int(df[col].nunique(dropna=True)),
            "sample": ", ".join(df[col].dropna().astype(str).head(3).tolist())[:100],
        })

    return {
        "rows": int(rows), "columns": int(columns),
        "duplicate_rows": duplicate_rows,
        "missing_cells": int(missing.sum()),
        "missing_pct": round(float(missing.sum() / max(rows * columns, 1) * 100), 2),
        "constant_columns": constant_columns,
        "date_columns": date_columns,
        "identifier_like": identifier_like,
        "high_cardinality": high_cardinality,
        "suspicious_columns": suspicious,
        "warnings": warnings,
        "column_profile": pd.DataFrame(column_rows),
        "target_summary": target_summary,
    }
