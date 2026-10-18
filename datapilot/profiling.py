"""Dataset profiling and transparent heuristic warnings for DataPilot AI."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from .leakage import detect_target_leakage


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


def _detect_corrupted_numeric(series: pd.Series, col_name: str) -> dict[str, Any] | None:
    """Detect columns that appear numeric but contain symbols, units, or non-numeric strings."""
    if pd.api.types.is_numeric_dtype(series):
        return None
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return None

    clean_s = series.dropna().astype(str).str.strip()
    if len(clean_s) < 5 or clean_s.nunique() < 3:
        return None

    # If already clean numeric strings, don't flag as corrupted
    direct_numeric = pd.to_numeric(clean_s, errors="coerce")
    if direct_numeric.notna().mean() == 1.0:
        return None

    # Check for currency, percentage, units, separators
    has_currency = bool(clean_s.str.contains(r"[$€£¥]", regex=True).mean() > 0.25)
    has_pct = bool(clean_s.str.contains(r"%", regex=True).mean() > 0.25)
    has_units = bool(
        clean_s.str.contains(
            r"(?i)(?:kg|g|lbs|m|km|cm|ml|l|mph|kph|hz|sec|ms)|\d+\s*(?:kg|g|lbs|m|km|cm|ml|l|mph|kph|hz|sec|ms)$",
            regex=True,
        ).mean()
        > 0.25
    )
    has_separators = bool(
        clean_s.str.contains(r"^\s*[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*$", regex=True).mean() > 0.25
    )

    cleaned = clean_s.str.replace(r"[^\d.\-]", "", regex=True)
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return None

    num_parsed = pd.to_numeric(cleaned, errors="coerce")
    clean_rate = float(num_parsed.notna().sum()) / max(len(clean_s), 1)

    if clean_rate >= 0.70:
        if has_currency:
            pattern = "currency symbols ($/€/£)"
        elif has_pct:
            pattern = "percentage symbols (%)"
        elif has_units:
            pattern = "measurement units (e.g. kg, m)"
        elif has_separators:
            pattern = "number separators (e.g. commas)"
        else:
            pattern = "mixed non-numeric string artifacts"

        corrupted_count = int((~direct_numeric.notna()).sum())
        sample_values = [str(v)[:40] for v in clean_s.loc[~direct_numeric.notna()].head(3).tolist()]
        if not sample_values:
            sample_values = [str(v)[:40] for v in clean_s.head(3).tolist()]

        return {
            "column": str(col_name),
            "clean_rate": round(clean_rate * 100, 1),
            "corrupted_count": corrupted_count,
            "detected_pattern": pattern,
            "sample_values": sample_values,
            "explanation": (
                f"Column '{col_name}' contains values formatted with {pattern} "
                f"({clean_rate:.1%} parseable as numeric). Currently treated as categorical text."
            ),
        }
    return None


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


def profile_dataframe(
    df: pd.DataFrame,
    target: str | None = None,
    time_col: str | None = None,
) -> dict[str, Any]:
    if df.empty:
        raise ValueError("The dataset has no rows.")

    rows, columns = df.shape
    missing = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())
    constant_columns = [str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1]
    identifier_like: list[str] = []
    high_cardinality: list[str] = []
    date_columns: list[str] = []
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    corrupted_numeric: list[dict[str, Any]] = []

    for col in df.columns:
        series = df[col]
        col_str = str(col)
        name = col_str.lower()
        if _is_date_like(series, col_str) or pd.api.types.is_datetime64_any_dtype(series):
            date_columns.append(col_str)
        elif pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_columns.append(col_str)
        else:
            categorical_columns.append(col_str)

        if re.search(r"(^|[_\W])(id|uuid|email|phone|customer_id|user_id)([_\W]|$)", name) and series.nunique(dropna=True) / max(rows, 1) > 0.5:
            identifier_like.append(col_str)
        if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype)) and series.nunique(dropna=True) > min(50, max(20, int(rows * 0.5))):
            high_cardinality.append(col_str)

        # Check for corrupted numeric representations
        cn = _detect_corrupted_numeric(series, col_str)
        if cn:
            corrupted_numeric.append(cn)

    warnings: list[dict[str, str]] = []
    if duplicate_rows:
        warnings.append({
            "level": "warning",
            "title": "Duplicate rows found",
            "detail": f"{duplicate_rows:,} exact duplicate rows ({duplicate_rows / max(rows, 1):.1%}). Confirm whether these are repeated events or accidental duplicates.",
        })
    missing_cols = {str(c): int(v) for c, v in missing.items() if v > 0}
    if missing_cols:
        top = sorted(missing_cols.items(), key=lambda item: item[1], reverse=True)[:4]
        preview = ", ".join(f"{name} ({count:,})" for name, count in top)
        warnings.append({
            "level": "warning",
            "title": "Missing values detected",
            "detail": f"Missing values occur in {len(missing_cols)} column(s). Top columns: {preview}. Imputation must be learned on the training split only.",
        })
    if constant_columns:
        warnings.append({
            "level": "info",
            "title": "Constant columns",
            "detail": "These columns contain one distinct value and probably add no predictive signal: " + ", ".join(constant_columns[:8]) + ("…" if len(constant_columns) > 8 else "."),
        })
    if identifier_like:
        warnings.append({
            "level": "warning",
            "title": "Possible identifier leakage",
            "detail": "High-cardinality identifier-like columns: " + ", ".join(identifier_like[:8]) + ". IDs can encourage memorization or reveal row identity; review before modeling.",
        })
    if high_cardinality:
        warnings.append({
            "level": "warning",
            "title": "High-cardinality text columns",
            "detail": "Potentially expensive or unhelpful columns: " + ", ".join(high_cardinality[:8]) + ". Consider dropping, grouping, or deliberately engineering these features.",
        })
    if corrupted_numeric:
        for item in corrupted_numeric:
            warnings.append({
                "level": "warning",
                "title": "Corrupted numeric column detected",
                "detail": (
                    f"Column '{item['column']}' appears numeric ({item['clean_rate']:.1f}% parseable) "
                    f"but contains {item['detected_pattern']}. It is currently treated as categorical text. "
                    "Clean non-numeric characters to model as continuous numeric."
                ),
            })
    if rows < 100:
        warnings.append({
            "level": "warning",
            "title": "Small dataset",
            "detail": f"Only {rows} rows are available. A single hold-out result may be unstable; treat metrics as exploratory.",
        })
    if rows < 30:
        warnings.append({
            "level": "error",
            "title": "Very small dataset",
            "detail": "Fewer than 30 rows are available. DataPilot will block model evaluation by default because a hold-out comparison is unlikely to be meaningful.",
        })

    # Explainable target leakage detection
    leakage_findings: list[dict[str, Any]] = []
    suspicious: list[str] = []
    if target and target in df.columns:
        raw_findings = detect_target_leakage(df, target, time_col=time_col)
        leakage_findings = [f.to_dict() for f in raw_findings]
        for f in raw_findings:
            if f.feature_name not in suspicious:
                suspicious.append(f.feature_name)

        critical_copies = [f.feature_name for f in raw_findings if f.finding_category == "confirmed_direct_leakage"]
        proxies = [f.feature_name for f in raw_findings if f.finding_category == "suspected_target_proxy"]
        suspicious_names = [f.feature_name for f in raw_findings if f.finding_category == "suspicious_feature_name"]
        temporal_leakage = [f.feature_name for f in raw_findings if f.finding_category == "possible_temporal_leakage"]

        if critical_copies:
            warnings.append({
                "level": "error",
                "title": "Critical target leakage detected",
                "detail": f"Direct target copies found: {', '.join(critical_copies)}. These features directly duplicate the target and must be excluded from modeling.",
            })
        if proxies:
            warnings.append({
                "level": "warning",
                "title": "Suspected target proxy detected",
                "detail": f"Features with near-perfect target association: {', '.join(proxies)}. Review whether these represent post-event data.",
            })
        if suspicious_names:
            warnings.append({
                "level": "warning",
                "title": "Potential target leakage",
                "detail": "Review these columns carefully: " + ", ".join(suspicious_names[:10]) + ". This is a name-based heuristic, not proof of leakage. Remove features that would not exist at prediction time.",
            })
        if temporal_leakage:
            warnings.append({
                "level": "warning",
                "title": "Possible temporal leakage",
                "detail": f"Features with timestamps after target timestamp: {', '.join(temporal_leakage)}. Review chronological ordering.",
            })

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

    memory_bytes = int(df.memory_usage(deep=True).sum())
    memory_mb = round(float(memory_bytes / (1024 * 1024)), 2)

    suggested_exclusions: list[str] = []
    for c in constant_columns:
        if c != target and c not in suggested_exclusions:
            suggested_exclusions.append(c)
    for c in suspicious:
        if c != target and c not in suggested_exclusions:
            suggested_exclusions.append(c)

    return {
        "rows": int(rows),
        "columns": int(columns),
        "duplicate_rows": duplicate_rows,
        "missing_cells": int(missing.sum()),
        "missing_pct": round(float(missing.sum() / max(rows * columns, 1) * 100), 2),
        "constant_columns": constant_columns,
        "date_columns": date_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "identifier_like": identifier_like,
        "high_cardinality": high_cardinality,
        "high_cardinality_columns": high_cardinality,
        "corrupted_numeric": corrupted_numeric,
        "suspicious_columns": suspicious,
        "suggested_exclusions": suggested_exclusions,
        "leakage_findings": leakage_findings,
        "warnings": warnings,
        "column_profile": pd.DataFrame(column_rows),
        "target_summary": target_summary,
        "memory_bytes": memory_bytes,
        "memory_mb": memory_mb,
    }
