"""Exportable human-readable audit reports."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .evaluation import EvaluationBundle, build_decision


def build_markdown_report(
    dataset_name: str, profile: dict[str, Any], bundle: EvaluationBundle
) -> str:
    decision = build_decision(bundle)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# DataPilot AI — Prediction Readiness Report",
        "",
        f"- **Generated:** {now}",
        f"- **Dataset:** {dataset_name}",
        f"- **Total rows / columns:** {profile['rows']:,} / {profile['columns']:,}",
    ]

    if bundle.dropped_target_rows > 0:
        lines.append(
            f"- **Eligible rows evaluated:** {bundle.eligible_rows:,} "
            f"({bundle.dropped_target_rows:,} missing-target rows excluded)"
        )

    lines.extend([
        f"- **Target:** `{bundle.target}`",
        f"- **Task:** {bundle.task.title()}",
        f"- **Validation:** {bundle.split_strategy}",
        f"- **Train / held-out rows:** {bundle.train_rows:,} / {bundle.test_rows:,}",
        f"- **Excluded features ({len(bundle.excluded_columns)}):** "
        + (", ".join(f"`{c}`" for c in bundle.excluded_columns) if bundle.excluded_columns else "None"),
        f"- **Active model features ({len(bundle.feature_columns)}):** "
        + ", ".join(f"`{c}`" for c in bundle.feature_columns),
        "",
        "## Decision summary",
        "",
        f"**{decision['label']}** — {decision['reason']}",
        "",
        "## Model metrics",
        "",
    ])

    metrics_df = pd.DataFrame(bundle.metrics).T
    lines.append(
        metrics_df.to_markdown(floatfmt=".4f")
        if not metrics_df.empty
        else "No metrics available."
    )

    lines.extend([
        "",
        "## Target leakage audit",
        "",
    ])

    leakage_findings = profile.get("leakage_findings", [])
    if leakage_findings:
        for finding in leakage_findings:
            feat = finding["feature_name"]
            cat = finding["finding_category"].replace("_", " ").title()
            sev = finding["severity"].upper()
            is_excluded = feat in bundle.excluded_columns
            status_badge = "✅ **EXCLUDED FROM MODELS**" if is_excluded else "🚨 **ACTIVE IN MODEL INPUTS**"

            lines.append(f"### Feature `{feat}` — {cat} ({sev})")
            lines.append(f"- **Status:** {status_badge}")
            lines.append(f"- **Explanation:** {finding['explanation']}")
            lines.append(f"- **Evidence:** {finding['evidence']}")
            lines.append(f"- **Recommended action:** {finding['recommended_action']}")
            lines.append("")
    else:
        lines.append(
            "No direct target copies, suspected proxies, or suspicious feature names were detected by automated checks. "
            "Always inspect domain context for temporal and operational availability."
        )
        lines.append("")

    lines.extend([
        "## Dataset quality",
        "",
        f"- Duplicate rows: {profile['duplicate_rows']:,}",
        f"- Missing cells: {profile['missing_cells']:,} ({profile['missing_pct']:.2f}%)",
        f"- Constant columns: {', '.join(profile['constant_columns']) or 'None detected'}",
        "",
        "## Warnings",
        "",
    ])

    all_warnings = [
        f"{item['title']}: {item['detail']}" for item in profile["warnings"]
    ]
    all_warnings.extend(bundle.warnings)
    all_warnings = list(dict.fromkeys(all_warnings))
    if all_warnings:
        lines.extend([f"- {warning}" for warning in all_warnings])
    else:
        lines.append(
            "No warnings were emitted by the implemented checks. This does not mean the dataset has no issues."
        )

    lines.extend([
        "",
        "## Limitations",
        "",
        "- This is an exploratory evaluation, not a guarantee of future performance.",
        "- A name-based leakage warning is a heuristic and cannot prove that leakage exists or does not exist.",
        "- Random hold-out results can be optimistic when data is time-dependent or grouped. Use chronological or group-aware validation where appropriate.",
        "- Hosted TabPFN inference sends relevant dataset values to the TabPFN service. Do not upload confidential or personal data unless you have permission and have reviewed applicable policies.",
        "- Confirm the model and API terms before commercial use.",
        "",
        "## Reproducibility",
        "",
        "- Random seed: 42 for the implemented random split and baseline estimators.",
        f"- TabPFN model version: {bundle.tabpfn_version or 'v3.5 (default pinned)'}",
        f"- Models evaluated: {', '.join(bundle.model_status.keys())}",
        "- TabPFN status: " + bundle.model_status.get("TabPFN", "unknown"),
        "",
        "Generated with DataPilot AI.",
    ])
    return "\n".join(lines)
