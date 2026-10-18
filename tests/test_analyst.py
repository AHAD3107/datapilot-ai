"""Tests for AI Analyst provider adapters, Privacy Mode, and executive summary generation."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from datapilot.analyst import (
    DeterministicLocalAnalyst,
    GeminiAnalystClient,
    MockAnalystClient,
    _sanitize_error_message,
    get_analyst_client,
)
from datapilot.context import build_audit_context
from datapilot.evaluation import evaluate_dataset
from datapilot.policy import is_privacy_mode_enabled, set_privacy_mode
from datapilot.profiling import profile_dataframe
from datapilot.report import generate_executive_summary
from datapilot.sample_data import make_sample_sales, make_target_leakage_fixture


@pytest.fixture(autouse=True)
def reset_policy():
    """Ensure privacy mode override is reset after each test."""
    set_privacy_mode(None)
    yield
    set_privacy_mode(None)


def test_missing_credentials_handled_gracefully():
    with patch.dict(os.environ, {"AI_API_KEY": "", "GEMINI_API_KEY": ""}, clear=False):
        client = GeminiAnalystClient()
        assert not client.is_configured()

        df = make_sample_sales(weeks=10, skus=4)
        prof = profile_dataframe(df, target="units_next_week")
        bundle = evaluate_dataset(df, target="units_next_week", task="regression", run_tabpfn=False)
        ctx = build_audit_context(prof, bundle, "sales.csv")

        # Must not raise an exception; should return informative fallback
        resp = client.explain_audit(ctx, "What was the MAE?")
        assert "not configured" in resp.lower() or "deterministic" in resp.lower()


def test_privacy_mode_blocks_gemini_calls():
    set_privacy_mode(True)
    client = GeminiAnalystClient(api_key="mock_key_123")

    df = make_sample_sales(weeks=10, skus=4)
    prof = profile_dataframe(df, target="units_next_week")
    bundle = evaluate_dataset(df, target="units_next_week", task="regression", run_tabpfn=False)
    ctx = build_audit_context(prof, bundle, "sales.csv")

    resp = client.explain_audit(ctx, "Explain the baseline model")
    assert "Privacy Mode" in resp
    assert "disabled" in resp.lower()


def test_successful_mocked_gemini_inference():
    set_privacy_mode(False)
    mock_response = MagicMock()
    mock_response.text = "Grounded response: Random Forest achieved lower MAE than naive baseline."

    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = mock_response

    with patch.dict(os.environ, {"AI_API_KEY": "valid_mock_key_xyz"}, clear=False):
        client = GeminiAnalystClient()
        client._client = mock_client_instance

        df = make_sample_sales(weeks=10, skus=4)
        prof = profile_dataframe(df, target="units_next_week")
        bundle = evaluate_dataset(df, target="units_next_week", task="regression", run_tabpfn=False)
        ctx = build_audit_context(prof, bundle, "sales.csv")

        resp = client.explain_audit(ctx, "How did Random Forest perform?")
        assert resp == "Grounded response: Random Forest achieved lower MAE than naive baseline."
        mock_client_instance.models.generate_content.assert_called_once()


def test_api_timeout_error_falls_back_without_crash():
    set_privacy_mode(False)
    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.side_effect = TimeoutError("Request timed out after 30s")

    client = GeminiAnalystClient(api_key="mock_key_abc")
    client._client = mock_client_instance

    df = make_sample_sales(weeks=10, skus=4)
    prof = profile_dataframe(df, target="units_next_week")
    bundle = evaluate_dataset(df, target="units_next_week", task="regression", run_tabpfn=False)
    ctx = build_audit_context(prof, bundle, "sales.csv")

    resp = client.explain_audit(ctx, "Explain the results")
    assert "AI Analyst error" in resp
    assert "TimeoutError" in resp
    # Deterministic fallback is included
    assert "Deterministic Summary" in resp or "Model Performance Summary" in resp


def test_token_redacted_from_error_messages():
    secret_key = "AIzaSySuperSecretKey123456"
    with patch.dict(os.environ, {"AI_API_KEY": secret_key}, clear=False):
        raw_error = f"Error 401 using key {secret_key} on endpoint https://api.google.com?key={secret_key}"
        sanitized = _sanitize_error_message(raw_error)
        assert secret_key not in sanitized
        assert "[REDACTED_API_KEY]" in sanitized


def test_deterministic_local_analyst_answers_metrics_and_leakage():
    df = make_target_leakage_fixture(seed=42, n_rows=160)
    prof = profile_dataframe(df, target="will_renew")
    bundle = evaluate_dataset(df, target="will_renew", task="classification", run_tabpfn=False)
    ctx = build_audit_context(prof, bundle, "churn_challenge.csv")

    local = DeterministicLocalAnalyst()
    leak_resp = local.explain_audit(ctx, "Explain leakage findings")
    assert "Target Leakage Audit" in leak_resp
    assert "renewal_decision_recorded" in leak_resp
    assert "Confirmed Direct Leakage" in leak_resp

    metric_resp = local.explain_audit(ctx, "What are the model metrics?")
    assert "Model Performance Summary" in metric_resp
    assert "Random forest" in metric_resp


def test_executive_summary_contains_core_audit_pillars():
    df = make_sample_sales(weeks=10, skus=4)
    prof = profile_dataframe(df, target="units_next_week")
    bundle = evaluate_dataset(df, target="units_next_week", task="regression", run_tabpfn=False)

    summary = generate_executive_summary(prof, bundle, "sample_sales.csv", client=DeterministicLocalAnalyst())
    assert "Executive Summary" in summary
    assert "Dataset & Task Overview" in summary
    assert "Data Quality & Preprocessing" in summary
    assert "Target Leakage & Integrity Audit" in summary
    assert "Model Performance & Evaluation" in summary
    assert "Decision Verdict" in summary
    assert "Recommended Next Steps" in summary
    assert "units_next_week" in summary


def test_factory_returns_mock_client():
    client = get_analyst_client("mock")
    assert isinstance(client, MockAnalystClient)
