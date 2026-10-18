"""Target leakage and temporal leakage detection for DataPilot AI."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class LeakageFinding:
    feature_name: str
    finding_category: str  # confirmed_direct_leakage, suspected_target_proxy, suspicious_feature_name, possible_temporal_leakage
    severity: str  # critical, high, warning, info
    explanation: str
    evidence: str
    recommended_action: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


_SUSPICIOUS_TERMS = re.compile(
    r"(^|[_\W])(future|next|post|after|outcome|result|target|label|actual|final|"
    r"returned|refund_issued|resolved|decision|recorded|renewed|churned|"
    r"cancelled|canceled|subsequent|closing|converted|purchased|settled)([_\W]|$)",
    re.I,
)


def _stem_tokens(text: str) -> list[str]:
    parts = re.split(r"[_\W]+", text.lower())
    stems = []
    for p in parts:
        if len(p) < 3:
            continue
        stems.append(p)
        if p.endswith("ing") and len(p) > 5:
            stems.append(p[:-3])
        elif p.endswith("ed") and len(p) > 4:
            stems.append(p[:-2])
        elif p.endswith("s") and len(p) > 3:
            stems.append(p[:-1])
    return list(dict.fromkeys(stems))


def detect_target_leakage(
    df: pd.DataFrame,
    target: str,
    time_col: str | None = None,
) -> list[LeakageFinding]:
    """Analyze features against target to identify confirmed and suspected leakage.

    Checks:
    1. Exact target copy (confirmed_direct_leakage).
    2. Binary inverse or equivalent encoding (confirmed_direct_leakage).
    3. Potential target proxy with near-perfect correlation/determination (suspected_target_proxy).
    4. Suspicious feature names indicating post-event/outcome information (suspicious_feature_name).
    5. Temporal leakage risk where timestamps/dates occur after reference time (possible_temporal_leakage).
    """
    if target not in df.columns:
        return []

    findings: list[LeakageFinding] = []
    y = df[target]
    rows = len(df)
    target_stems = _stem_tokens(target)

    for col in df.columns:
        if col == target:
            continue
        col_str = str(col)
        s = df[col]

        valid_mask = s.notna() & y.notna()
        n_valid = int(valid_mask.sum())

        has_direct_copy = False
        has_inverse_copy = False

        # --- 1. Exact Target Copy Check ---
        if n_valid >= min(10, rows):
            s_val = s.loc[valid_mask]
            y_val = y.loc[valid_mask]

            is_exact = False
            if (s_val.to_numpy() == y_val.to_numpy()).all():
                is_exact = True
            elif pd.api.types.is_numeric_dtype(s_val) and pd.api.types.is_numeric_dtype(y_val):
                if np.isclose(s_val.to_numpy(), y_val.to_numpy(), equal_nan=False).all():
                    is_exact = True
            elif s_val.astype(str).str.strip().str.lower().equals(y_val.astype(str).str.strip().str.lower()):
                is_exact = True

            if is_exact:
                has_direct_copy = True
                findings.append(
                    LeakageFinding(
                        feature_name=col_str,
                        finding_category="confirmed_direct_leakage",
                        severity="critical",
                        explanation=f"Feature '{col_str}' exactly reproduces target '{target}' for all non-missing rows.",
                        evidence=f"100.0% of non-missing values ({n_valid:,} rows) match target values exactly.",
                        recommended_action="Exclude this feature from all model training and evaluation. It duplicates the target.",
                    )
                )

        # --- 2. Binary Inverse / Inverted Encoding Check ---
        if not has_direct_copy and n_valid >= min(10, rows):
            s_val = s.loc[valid_mask]
            y_val = y.loc[valid_mask]

            s_uniq = s_val.unique()
            y_uniq = y_val.unique()

            if len(s_uniq) == 2 and len(y_uniq) == 2:
                is_inverse = False
                inv_evidence = ""
                if pd.api.types.is_numeric_dtype(s_val) and pd.api.types.is_numeric_dtype(y_val):
                    set_s = set(s_uniq)
                    set_y = set(y_uniq)
                    if set_s == {0, 1} and set_y == {0, 1}:
                        if (s_val.to_numpy() == (1 - y_val.to_numpy())).all():
                            is_inverse = True
                            inv_evidence = f"Exact inverted binary encoding (1 - target) across all {n_valid:,} rows."

                if not is_inverse:
                    ct = pd.crosstab(s_val, y_val)
                    if ct.shape == (2, 2):
                        arr = ct.to_numpy()
                        if arr[0, 0] == 0 and arr[1, 1] == 0:
                            is_inverse = True
                            inv_evidence = (
                                f"Perfect 1-to-1 inverted mapping ({ct.index[0]} -> {ct.columns[1]}, "
                                f"{ct.index[1]} -> {ct.columns[0]}) across {n_valid:,} rows."
                            )

                if is_inverse:
                    has_inverse_copy = True
                    findings.append(
                        LeakageFinding(
                            feature_name=col_str,
                            finding_category="confirmed_direct_leakage",
                            severity="critical",
                            explanation=f"Feature '{col_str}' is an inverted binary encoding of target '{target}'.",
                            evidence=inv_evidence,
                            recommended_action="Exclude this feature from all model training and evaluation. It is an inverted copy of the target.",
                        )
                    )

        # --- 3. Potential Target Proxy Check ---
        if not has_direct_copy and not has_inverse_copy and n_valid >= 30:
            s_val = s.loc[valid_mask]
            y_val = y.loc[valid_mask]

            if pd.api.types.is_numeric_dtype(s_val) and pd.api.types.is_numeric_dtype(y_val):
                std_s = float(s_val.std())
                std_y = float(y_val.std())
                if std_s > 1e-9 and std_y > 1e-9:
                    corr = float(np.corrcoef(s_val.to_numpy(), y_val.to_numpy())[0, 1])
                    if not np.isnan(corr) and abs(corr) >= 0.999:
                        findings.append(
                            LeakageFinding(
                                feature_name=col_str,
                                finding_category="suspected_target_proxy",
                                severity="high",
                                explanation=f"Feature '{col_str}' has near-perfect linear correlation ({corr:.4f}) with target '{target}'.",
                                evidence=f"Pearson correlation coefficient is {corr:.4f} across {n_valid:,} observations.",
                                recommended_action="Review feature lineage. Features with near-perfect correlation are frequently post-event proxies.",
                            )
                        )
            elif y_val.nunique() <= 10 and s_val.nunique() <= 50:
                ct = pd.crosstab(s_val, y_val)
                if (ct > 0).sum(axis=1).max() == 1 and s_val.nunique() >= y_val.nunique():
                    findings.append(
                        LeakageFinding(
                            feature_name=col_str,
                            finding_category="suspected_target_proxy",
                            severity="high",
                            explanation=f"Feature '{col_str}' deterministically partitions target '{target}' with 100% conditional purity.",
                            evidence=f"Every category of '{col_str}' maps to exactly one target class across {n_valid:,} rows.",
                            recommended_action="Review whether this column is created after the outcome is decided.",
                        )
                    )

        # --- 4. Suspicious Feature Name Check ---
        col_lower = col_str.lower()
        has_suspicious_term = bool(_SUSPICIOUS_TERMS.search(col_lower))
        has_target_stem_combo = False
        stem_combo_detail = ""

        if target_stems:
            matched_stems = [stem for stem in target_stems if stem in col_lower]
            if matched_stems:
                context_terms = (
                    "decision", "recorded", "status", "flag", "date",
                    "result", "outcome", "after", "post", "actual", "final"
                )
                found_context = [ctx for ctx in context_terms if ctx in col_lower]
                if found_context:
                    has_target_stem_combo = True
                    stem_combo_detail = f"target stem '{matched_stems[0]}' combined with outcome descriptor '{found_context[0]}'"

        if has_suspicious_term or has_target_stem_combo:
            term_desc = stem_combo_detail if has_target_stem_combo else "naming pattern suggesting post-event or outcome data"
            existing_categories = [f.finding_category for f in findings if f.feature_name == col_str]
            if "confirmed_direct_leakage" not in existing_categories:
                findings.append(
                    LeakageFinding(
                        feature_name=col_str,
                        finding_category="suspicious_feature_name",
                        severity="warning",
                        explanation=f"Feature name '{col_str}' contains a {term_desc}.",
                        evidence=f"Matched suspicious name pattern in '{col_str}'.",
                        recommended_action="Check when this field is recorded in business workflows. Exclude if unavailable at prediction time.",
                    )
                )

        # --- 5. Possible Temporal Leakage Check ---
        if time_col and time_col in df.columns and col_str != time_col:
            if pd.api.types.is_datetime64_any_dtype(s) or any(
                token in col_lower for token in ("date", "time", "timestamp", "completed", "closed", "resolved", "ended")
            ):
                t_dates = pd.to_datetime(df[time_col], errors="coerce", format="mixed")
                c_dates = pd.to_datetime(s, errors="coerce", format="mixed")
                valid_dates = t_dates.notna() & c_dates.notna()
                if valid_dates.sum() >= 10:
                    pct_after = float((c_dates.loc[valid_dates] > t_dates.loc[valid_dates]).mean())
                    if pct_after > 0.05:
                        findings.append(
                            LeakageFinding(
                                feature_name=col_str,
                                finding_category="possible_temporal_leakage",
                                severity="warning",
                                explanation=f"Feature '{col_str}' contains timestamps occurring after the prediction reference time '{time_col}'.",
                                evidence=f"{pct_after:.1%} of observed dates in '{col_str}' occur after '{time_col}'.",
                                recommended_action=f"Exclude '{col_str}' or restrict it to observations strictly preceding '{time_col}'.",
                            )
                        )

    return findings
