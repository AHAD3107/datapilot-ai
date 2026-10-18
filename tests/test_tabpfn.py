"""Tests for TabPFN resilience, mock inference, error handling, and credential safety."""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from datapilot.evaluation import _sanitize_error_message, evaluate_dataset
from datapilot.sample_data import make_sample_sales


def test_missing_token_sets_not_configured_and_preserves_local_baselines():
    df = make_sample_sales(weeks=10, skus=4)
    with patch.dict(os.environ, {"TABPFN_TOKEN": ""}, clear=False):
        bundle = evaluate_dataset(df, target="units_next_week", task="regression", run_tabpfn=True)
        assert bundle.model_status["TabPFN"] == "not configured"
        assert "Naive baseline" in bundle.metrics
        assert "Random forest" in bundle.metrics
        assert bundle.model_status["Naive baseline"] == "completed"
        assert bundle.model_status["Random forest"] == "completed"
        assert any("TABPFN_TOKEN is not configured" in w for w in bundle.warnings)


def test_sanitize_error_message_redacts_tokens_and_secrets():
    secret_token = "tb_live_secret_token_9876543210"
    raw_error = f"Unauthorized error: 401 using {secret_token} on https://user:{secret_token}@api.tabpfn.com"
    clean = _sanitize_error_message(raw_error, token=secret_token)
    assert secret_token not in clean
    assert "[REDACTED_TABPFN_TOKEN]" in clean


def test_successful_mocked_tabpfn_inference():
    df = make_sample_sales(weeks=10, skus=4)
    dummy_pred = np.full(shape=(bundle_test_len := 8,), fill_value=25.0)

    mock_model = MagicMock()
    mock_model.fit.return_value = None
    mock_model.predict.return_value = dummy_pred

    with patch.dict(os.environ, {"TABPFN_TOKEN": "valid_mock_token_abc123"}, clear=False):
        with patch("datapilot.evaluation.SimpleImputer"):
            pass  # keep normal
        with patch("tabpfn_client.TabPFNRegressor.create_default_for_version", return_value=mock_model):
            bundle = evaluate_dataset(df, target="units_next_week", task="regression", run_tabpfn=True)
            assert bundle.model_status["TabPFN"] == "completed"
            assert "TabPFN" in bundle.metrics
            assert "MAE" in bundle.metrics["TabPFN"]
            assert "TabPFN" in bundle.model_timings
            assert bundle.tabpfn_version == "v3.5"


def test_tabpfn_auth_failure_does_not_crash_application():
    df = make_sample_sales(weeks=10, skus=4)
    secret_token = "secret_auth_token_xyz999"

    mock_model = MagicMock()
    mock_model.fit.side_effect = PermissionError(f"401 Unauthorized for token {secret_token}")

    with patch.dict(os.environ, {"TABPFN_TOKEN": secret_token}, clear=False):
        with patch("tabpfn_client.TabPFNRegressor.create_default_for_version", return_value=mock_model):
            bundle = evaluate_dataset(df, target="units_next_week", task="regression", run_tabpfn=True)
            assert bundle.model_status["TabPFN"] == "failed"
            assert "TabPFN" not in bundle.metrics
            # Baselines still succeeded
            assert bundle.model_status["Random forest"] == "completed"
            assert "Random forest" in bundle.metrics
            # Token is never leaked into warnings
            all_warnings_str = " ".join(bundle.warnings)
            assert secret_token not in all_warnings_str
            assert "Authentication failed" in all_warnings_str


def test_tabpfn_timeout_error_is_classified_gracefully():
    df = make_sample_sales(weeks=10, skus=4)

    mock_model = MagicMock()
    mock_model.fit.side_effect = TimeoutError("Connection to tabpfn hosted service timed out after 30s")

    with patch.dict(os.environ, {"TABPFN_TOKEN": "mock_token_123"}, clear=False):
        with patch("tabpfn_client.TabPFNRegressor.create_default_for_version", return_value=mock_model):
            bundle = evaluate_dataset(df, target="units_next_week", task="regression", run_tabpfn=True)
            assert bundle.model_status["TabPFN"] == "failed"
            all_warnings_str = " ".join(bundle.warnings)
            assert "timed out" in all_warnings_str.lower()
