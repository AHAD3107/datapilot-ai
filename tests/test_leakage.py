"""Tests for target leakage detection and feature exclusion verification."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datapilot.evaluation import evaluate_dataset
from datapilot.leakage import detect_target_leakage
from datapilot.profiling import profile_dataframe
from datapilot.sample_data import make_target_leakage_fixture


def test_detect_exact_target_copy():
    df = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        "exact_copy": [10, 20, 30, 40, 50] * 10,
        "target": [10, 20, 30, 40, 50] * 10,
    })
    findings = detect_target_leakage(df, "target")
    categories = [f.finding_category for f in findings]
    features = [f.feature_name for f in findings]
    assert "exact_copy" in features
    copy_finding = next(f for f in findings if f.feature_name == "exact_copy")
    assert copy_finding.finding_category == "confirmed_direct_leakage"
    assert copy_finding.severity == "critical"
    assert "feature_a" not in [f.feature_name for f in findings if f.finding_category == "confirmed_direct_leakage"]


def test_detect_binary_inverse():
    df = pd.DataFrame({
        "feature_normal": [1, 2, 3, 4] * 10,
        "binary_inverse": [1, 0, 1, 0] * 10,
        "target": [0, 1, 0, 1] * 10,
    })
    findings = detect_target_leakage(df, "target")
    inverse_finding = next((f for f in findings if f.feature_name == "binary_inverse"), None)
    assert inverse_finding is not None
    assert inverse_finding.finding_category in {"confirmed_direct_leakage", "suspected_target_proxy"}


def test_suspicious_feature_names_flagged_with_caution():
    # Use varied targets so the categorical feature is not a 1-to-1 proxy
    df = pd.DataFrame({
        "post_decision_notes": ["standard note", "escalation"] * 20,
        "normal_feature": range(40),
        "target": range(100, 140),
    })
    findings = detect_target_leakage(df, "target")
    name_finding = next((f for f in findings if f.feature_name == "post_decision_notes"), None)
    assert name_finding is not None
    assert name_finding.finding_category == "suspicious_feature_name"
    assert "suggesting" in name_finding.explanation.lower()
    assert name_finding.severity in {"warning", "high", "medium"}


def test_normal_predictive_feature_not_falsely_labeled_direct_leakage():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(0, 1, n)
    target = x * 2.0 + rng.normal(0, 1, n)
    df = pd.DataFrame({"x": x, "target": target})
    findings = detect_target_leakage(df, "target")
    direct_leaks = [f for f in findings if f.finding_category == "confirmed_direct_leakage"]
    assert len(direct_leaks) == 0


def test_renewal_decision_recorded_challenge_leakage_detected():
    df = make_target_leakage_fixture(seed=42, n_rows=160)
    findings = detect_target_leakage(df, "will_renew")
    finding_map = {f.feature_name: f for f in findings}
    assert "renewal_decision_recorded" in finding_map
    assert finding_map["renewal_decision_recorded"].finding_category == "confirmed_direct_leakage"
    assert finding_map["renewal_decision_recorded"].severity == "critical"


def test_leakage_findings_integrated_in_profiling():
    df = make_target_leakage_fixture(seed=42, n_rows=160)
    profile = profile_dataframe(df, target="will_renew")
    assert "leakage_findings" in profile
    assert len(profile["leakage_findings"]) > 0
    assert "renewal_decision_recorded" in profile["suspicious_columns"]
    # Check that error-level warning was added for critical leakage
    critical_warnings = [w for w in profile["warnings"] if w["level"] == "error" and "leakage" in w["title"].lower()]
    assert len(critical_warnings) >= 1


def test_excluded_features_are_strictly_absent_from_model_inputs():
    df = make_target_leakage_fixture(seed=42, n_rows=160)
    excluded = ["renewal_decision_recorded", "post_decision_notes"]
    bundle = evaluate_dataset(
        df,
        target="will_renew",
        task="classification",
        excluded_columns=excluded,
        run_tabpfn=False,
    )
    assert "renewal_decision_recorded" not in bundle.feature_columns
    assert "post_decision_notes" not in bundle.feature_columns
    assert "renewal_decision_recorded" in bundle.excluded_columns
    assert "post_decision_notes" in bundle.excluded_columns
    assert set(bundle.feature_columns).isdisjoint(set(excluded))
