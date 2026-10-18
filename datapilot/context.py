"""Sanitized, serializable AuditContext DTO for grounded AI analysis."""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .evaluation import EvaluationBundle, build_decision


def _sanitize_string(val: Any, max_length: int = 120) -> str:
    """Sanitize untrusted strings (column names, categories) against prompt injection and control chars."""
    if val is None:
        return ""
    text = str(val).strip()
    # Strip potential token disclosures
    for secret_env in ("TABPFN_TOKEN", "AI_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        sec = os.getenv(secret_env, "").strip()
        if sec and len(sec) >= 6:
            text = text.replace(sec, "[REDACTED_SECRET]")

    # Normalize line breaks and tabs into spaces
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Remove non-printable control characters
    text = "".join(c for c in text if c.isprintable())
    # Limit maximum length
    if len(text) > max_length:
        text = text[:max_length] + "..."
    return text


@dataclass
class AuditContext:
    """Typed, bounded representation of a DataPilot audit for grounded explanations."""

    dataset_name: str
    total_rows: int
    total_columns: int
    eligible_rows: int
    dropped_target_rows: int
    target_name: str
    inferred_task: str
    selected_task: str
    split_strategy: str
    train_rows: int
    held_out_rows: int
    model_metrics: dict[str, dict[str, float | None]]
    model_statuses: dict[str, str]
    model_timings: dict[str, float]
    tabpfn_version: str | None
    excluded_features: list[str]
    active_features: list[str]
    missing_cells: int
    duplicate_rows: int
    constant_columns: list[str]
    high_cardinality_columns: list[str]
    corrupted_numeric_columns: list[dict[str, Any]]
    leakage_findings: list[dict[str, Any]]
    decision_label: str
    decision_reason: str
    warnings: list[str]
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary with honest null representations."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to deterministic formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_compact_prompt_summary(self) -> str:
        """Produce a compact, structured text summary for LLM grounding (zero raw CSV rows)."""
        lines = [
            f"=== DATAPILOT AUDIT CONTEXT (v{self.schema_version}) ===",
            f"Dataset: `{self.dataset_name}`",
            f"Dimensions: {self.total_rows:,} total rows ({self.dropped_target_rows} dropped null targets -> {self.eligible_rows:,} eligible rows), {self.total_columns} columns",
            f"Target: `{self.target_name}` | Task: {self.selected_task.title()} (Inferred: {self.inferred_task.title()})",
            f"Validation Split: {self.split_strategy} ({self.train_rows:,} train / {self.held_out_rows:,} held-out rows)",
            f"Excluded Features ({len(self.excluded_features)}): {', '.join(f'`{c}`' for c in self.excluded_features) if self.excluded_features else 'None'}",
            f"Active Features ({len(self.active_features)}): {', '.join(f'`{c}`' for c in self.active_features)}",
            "",
            "--- Model Evaluation Results ---",
        ]

        if not self.model_metrics:
            lines.append("No model metrics available.")
        else:
            for model_name, metrics in self.model_metrics.items():
                status = self.model_statuses.get(model_name, "unknown")
                timing = self.model_timings.get(model_name)
                timing_str = f" in {timing:.2f}s" if timing is not None else ""
                metrics_str = ", ".join(
                    f"{k}: {f'{v:.4f}' if isinstance(v, (int, float)) else 'N/A'}"
                    for k, v in metrics.items()
                )
                lines.append(f"- **{model_name}** (status: {status}{timing_str}): {metrics_str}")

        lines.extend([
            "",
            f"--- Decision Verdict: {self.decision_label} ---",
            f"Reason: {self.decision_reason}",
            "",
            "--- Data Quality & Integrity Findings ---",
            f"- Duplicate rows: {self.duplicate_rows:,}",
            f"- Missing cells: {self.missing_cells:,}",
            f"- Constant columns: {', '.join(self.constant_columns) if self.constant_columns else 'None'}",
            f"- High-cardinality text columns: {', '.join(self.high_cardinality_columns) if self.high_cardinality_columns else 'None'}",
        ])

        if self.corrupted_numeric_columns:
            lines.append(f"- Corrupted numeric columns ({len(self.corrupted_numeric_columns)}):")
            for c in self.corrupted_numeric_columns:
                lines.append(f"  * `{c['column']}`: {c.get('clean_rate', 0):.1f}% parseable, pattern: {c.get('detected_pattern', 'unknown')}")
        else:
            lines.append("- Corrupted numeric columns: None detected")

        if self.leakage_findings:
            lines.append(f"- Target leakage findings ({len(self.leakage_findings)}):")
            for f in self.leakage_findings:
                feat = f.get("feature_name", "unknown")
                is_excl = feat in self.excluded_features
                status_badge = "EXCLUDED FROM EVALUATION" if is_excl else "ACTIVE IN MODEL INPUTS"
                lines.append(f"  * Feature `{feat}` [{f.get('severity', '').upper()}] - {f.get('finding_category', '')} ({status_badge})")
                lines.append(f"    Evidence: {f.get('evidence', '')}")
                lines.append(f"    Action: {f.get('recommended_action', '')}")
        else:
            lines.append("- Target leakage findings: None detected")

        if self.warnings:
            lines.extend(["", "--- Audit Warnings & Caveats ---"])
            for w in self.warnings[:8]:
                lines.append(f"- {w}")

        lines.append("=== END AUDIT CONTEXT ===")
        return "\n".join(lines)


def build_audit_context(
    profile: dict[str, Any],
    bundle: EvaluationBundle,
    dataset_name: str = "dataset.csv",
) -> AuditContext:
    """Build a sanitized, grounded AuditContext from actual profile and evaluation bundle objects."""
    decision = build_decision(bundle)

    # Sanitize strings to avoid prompt injection in column names
    clean_dataset_name = _sanitize_string(dataset_name, max_length=80)
    clean_target = _sanitize_string(bundle.target, max_length=60)
    clean_excluded = [_sanitize_string(c, max_length=60) for c in bundle.excluded_columns]
    clean_active = [_sanitize_string(c, max_length=60) for c in bundle.feature_columns]
    clean_constants = [_sanitize_string(c, max_length=60) for c in profile.get("constant_columns", [])]
    clean_high_card = [_sanitize_string(c, max_length=60) for c in profile.get("high_cardinality", [])]

    # Clean model metrics - maintain None where undefined rather than inventing zeroes
    sanitized_metrics: dict[str, dict[str, float | None]] = {}
    for model_name, metrics in bundle.metrics.items():
        clean_model_name = _sanitize_string(model_name, max_length=40)
        sanitized_metrics[clean_model_name] = {}
        for metric_name, val in metrics.items():
            clean_metric_name = _sanitize_string(metric_name, max_length=40)
            if val is None or (isinstance(val, float) and (val != val or abs(val) == float("inf"))):
                sanitized_metrics[clean_model_name][clean_metric_name] = None
            else:
                sanitized_metrics[clean_model_name][clean_metric_name] = float(val)

    # Corrupted numeric findings
    corrupted_numeric = []
    for item in profile.get("corrupted_numeric", []):
        corrupted_numeric.append({
            "column": _sanitize_string(item.get("column", ""), max_length=60),
            "clean_rate": float(item.get("clean_rate", 0)),
            "corrupted_count": int(item.get("corrupted_count", 0)),
            "detected_pattern": _sanitize_string(item.get("detected_pattern", ""), max_length=60),
            "explanation": _sanitize_string(item.get("explanation", ""), max_length=160),
        })

    # Leakage findings
    clean_leakage = []
    for item in profile.get("leakage_findings", []):
        clean_leakage.append({
            "feature_name": _sanitize_string(item.get("feature_name", ""), max_length=60),
            "finding_category": _sanitize_string(item.get("finding_category", ""), max_length=40),
            "severity": _sanitize_string(item.get("severity", ""), max_length=20),
            "explanation": _sanitize_string(item.get("explanation", ""), max_length=200),
            "evidence": _sanitize_string(item.get("evidence", ""), max_length=160),
            "recommended_action": _sanitize_string(item.get("recommended_action", ""), max_length=160),
        })

    # Deduplicate and sanitize warnings
    raw_warnings = [f"{item['title']}: {item['detail']}" for item in profile.get("warnings", [])]
    raw_warnings.extend(bundle.warnings)
    clean_warnings = list(dict.fromkeys(_sanitize_string(w, max_length=240) for w in raw_warnings))

    target_summary = profile.get("target_summary") or {}
    inferred_task = str(target_summary.get("inferred_task", bundle.task))

    return AuditContext(
        dataset_name=clean_dataset_name,
        total_rows=int(bundle.total_rows or profile.get("rows", 0)),
        total_columns=int(profile.get("columns", 0)),
        eligible_rows=int(bundle.eligible_rows or bundle.train_rows + bundle.test_rows),
        dropped_target_rows=int(bundle.dropped_target_rows or 0),
        target_name=clean_target,
        inferred_task=inferred_task,
        selected_task=str(bundle.task),
        split_strategy=str(bundle.split_strategy),
        train_rows=int(bundle.train_rows),
        held_out_rows=int(bundle.test_rows),
        model_metrics=sanitized_metrics,
        model_statuses={_sanitize_string(k, 40): _sanitize_string(v, 40) for k, v in bundle.model_status.items()},
        model_timings={_sanitize_string(k, 40): float(v) for k, v in bundle.model_timings.items()},
        tabpfn_version=str(bundle.tabpfn_version) if bundle.tabpfn_version else None,
        excluded_features=clean_excluded,
        active_features=clean_active,
        missing_cells=int(profile.get("missing_cells", 0)),
        duplicate_rows=int(profile.get("duplicate_rows", 0)),
        constant_columns=clean_constants,
        high_cardinality_columns=clean_high_card,
        corrupted_numeric_columns=corrupted_numeric,
        leakage_findings=clean_leakage,
        decision_label=str(decision.get("label", "UNKNOWN")),
        decision_reason=_sanitize_string(decision.get("reason", ""), max_length=250),
        warnings=clean_warnings,
    )
