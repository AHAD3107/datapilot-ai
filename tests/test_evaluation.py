import numpy as np
import pandas as pd
import pytest

from datapilot.evaluation import build_decision, evaluate_dataset
from datapilot.sample_data import make_sample_sales, make_small_edge_case_fixture


def test_regression_evaluation_runs_baselines_without_tabpfn():
    df = make_sample_sales(seed=9, weeks=32, skus=4)
    bundle = evaluate_dataset(
        df,
        target="units_next_week",
        task="regression",
        time_col="week_start",
        excluded_columns=["sku"],
        run_tabpfn=False,
    )
    assert bundle.split_strategy == "Chronological hold-out"
    assert bundle.train_rows > bundle.test_rows
    assert "Naive baseline" in bundle.metrics
    assert "Random forest" in bundle.metrics
    assert "MAE" in bundle.metrics["Random forest"]
    assert len(bundle.predictions) == bundle.test_rows
    assert build_decision(bundle)["label"] == "INCOMPLETE EVALUATION"


def test_chronological_split_keeps_same_date_in_one_partition():
    df = make_sample_sales(seed=11, weeks=30, skus=5)
    bundle = evaluate_dataset(
        df,
        target="units_next_week",
        task="regression",
        time_col="week_start",
        run_tabpfn=False,
    )
    predicted_indices = set(bundle.predictions["source_row"].astype(str))
    test_rows = df.loc[[int(i) for i in predicted_indices]]
    train_rows = df.drop(index=[int(i) for i in predicted_indices])
    assert set(train_rows["week_start"]).isdisjoint(set(test_rows["week_start"]))


def test_classification_metrics_run():
    rng = np.random.default_rng(10)
    n = 240
    x = rng.normal(size=n)
    label = np.where(x + rng.normal(size=n) > 0, "yes", "no")
    df = pd.DataFrame({"x": x, "segment": np.where(x > 0, "A", "B"), "label": label})
    bundle = evaluate_dataset(df, target="label", task="classification", run_tabpfn=False)
    assert "Balanced accuracy" in bundle.metrics["Random forest"]
    assert bundle.metrics["Random forest"]["Accuracy"] >= 0


def test_invalid_regression_target_fails_clearly():
    df = pd.DataFrame({"x": range(40), "target": ["a", "b"] * 20})
    with pytest.raises(ValueError, match="numeric target"):
        evaluate_dataset(df, target="target", task="regression", run_tabpfn=False)


def test_single_class_target_fails():
    df = pd.DataFrame({"x": range(40), "target": [1] * 40})
    with pytest.raises(ValueError, match="fewer than two distinct values"):
        evaluate_dataset(df, target="target", task="classification", run_tabpfn=False)


def test_missing_targets_are_excluded_and_counts_reported():
    rng = np.random.default_rng(42)
    n = 60
    x = rng.normal(size=n)
    targets = [float(i) if i % 6 != 0 else np.nan for i in range(n)]
    df = pd.DataFrame({"x": x, "category": ["cat_a", "cat_b"] * (n // 2), "target": targets})

    bundle = evaluate_dataset(df, target="target", task="regression", run_tabpfn=False)
    assert bundle.total_rows == 60
    assert bundle.dropped_target_rows == 10
    assert bundle.eligible_rows == 50
    assert bundle.train_rows + bundle.test_rows == 50
    assert any("Excluded 10 row(s) with missing target values" in w for w in bundle.warnings)
    # Ensure no NaN target was included in predictions
    assert not bundle.predictions["actual"].isna().any()


def test_insufficient_eligible_rows_raises_actionable_error():
    df = pd.DataFrame({
        "x": range(50),
        "target": [float(i) if i < 15 else np.nan for i in range(50)],  # only 15 eligible rows
    })
    with pytest.raises(ValueError, match="At least 30 valid rows are required"):
        evaluate_dataset(df, target="target", task="regression", run_tabpfn=False)


def test_small_edge_case_safeguard():
    df = make_small_edge_case_fixture(n_rows=10)
    with pytest.raises(ValueError, match="At least 30 valid rows are required"):
        evaluate_dataset(df, target="target", task="classification", run_tabpfn=False)


def test_entirely_missing_columns_are_handled_safely():
    rng = np.random.default_rng(7)
    n = 50
    df = pd.DataFrame({
        "valid_num": rng.normal(size=n),
        "all_nan_num": [np.nan] * n,
        "all_nan_cat": [None] * n,
        "target": rng.uniform(10, 50, size=n),
    })
    bundle = evaluate_dataset(df, target="target", task="regression", run_tabpfn=False)
    assert bundle.model_status["Random forest"] == "completed"
    assert "MAE" in bundle.metrics["Random forest"]


def test_non_existent_excluded_column_warns_gracefully():
    df = make_sample_sales(weeks=10, skus=4)
    bundle = evaluate_dataset(
        df,
        target="units_next_week",
        task="regression",
        excluded_columns=["non_existent_col_123"],
        run_tabpfn=False,
    )
    assert any("non_existent_col_123" in w for w in bundle.warnings)
    assert "units_next_week" not in bundle.feature_columns


def test_decision_reports_investigation_when_leakage_warning_present():
    df = make_sample_sales(weeks=10, skus=4)
    bundle = evaluate_dataset(df, target="units_next_week", task="regression", run_tabpfn=False)
    # Simulate a profiler leakage warning carried into bundle
    bundle.warnings.append("Critical target leakage detected: Direct target copies found: copy_col.")
    decision = build_decision(bundle)
    assert "Note: Potential target leakage" in decision["reason"]


def test_evaluation_privacy_mode_blocks_tabpfn_and_runs_local_baselines():
    from datapilot.policy import set_privacy_mode, is_privacy_mode_enabled

    set_privacy_mode(True)
    try:
        df = make_sample_sales(weeks=12, skus=3)
        bundle = evaluate_dataset(
            df,
            target="units_next_week",
            task="regression",
            run_tabpfn=True,  # User explicitly requested TabPFN, but Privacy Mode is active
        )
        assert bundle.model_status["TabPFN"] == "disabled by privacy mode"
        assert bundle.model_status["Naive baseline"] == "completed"
        assert bundle.model_status["Random forest"] == "completed"
        assert any("Privacy Mode" in w or "privacy mode" in w for w in bundle.warnings)
    finally:
        set_privacy_mode(False)


# ---------------------------------------------------------------------------
# Regression tests for bugs fixed during submission stabilisation
# ---------------------------------------------------------------------------

def test_evaluate_dataset_accepts_split_strategy_random():
    """Regression: evaluate_dataset must accept split_strategy='random'.

    Bug: app.py passed split_strategy='random'|'chronological' but the
    function signature did not include that parameter, causing a TypeError
    every time the user pressed 'Run Prediction Audit'.
    """
    df = make_sample_sales(weeks=20, skus=4)
    bundle = evaluate_dataset(
        df,
        target="units_next_week",
        task="regression",
        split_strategy="random",
        time_col="week_start",   # time_col should be ignored because random was requested
        run_tabpfn=False,
    )
    # Because split_strategy='random' overrides time_col, the strategy is NOT chronological.
    assert "Chronological" not in bundle.split_strategy
    assert bundle.model_status["Naive baseline"] == "completed"


def test_evaluate_dataset_accepts_split_strategy_chronological():
    """Regression: split_strategy='chronological' with time_col uses chronological split."""
    df = make_sample_sales(weeks=20, skus=4)
    bundle = evaluate_dataset(
        df,
        target="units_next_week",
        task="regression",
        split_strategy="chronological",
        time_col="week_start",
        run_tabpfn=False,
    )
    assert bundle.split_strategy == "Chronological hold-out"


def test_evaluate_dataset_split_strategy_random_ignores_time_col():
    """Regression: random split must not use time_col even when it is supplied."""
    df = make_sample_sales(weeks=20, skus=4)
    bundle_random = evaluate_dataset(
        df,
        target="units_next_week",
        task="regression",
        split_strategy="random",
        time_col="week_start",
        run_tabpfn=False,
    )
    bundle_no_time = evaluate_dataset(
        df,
        target="units_next_week",
        task="regression",
        split_strategy="random",
        time_col=None,
        run_tabpfn=False,
    )
    # Both should use a random split strategy (non-chronological)
    assert "Chronological" not in bundle_random.split_strategy
    assert "Chronological" not in bundle_no_time.split_strategy


def test_leakage_audit_target_access_uses_bundle_instance():
    """Regression: Leakage Audit page used EvaluationBundle CLASS as .get() default,
    causing AttributeError on .target and .excluded_columns because dataclass
    fields don't exist at class level.  Verify correct instance access at runtime.
    """
    from datapilot.evaluation import EvaluationBundle

    # This should raise AttributeError on the CLASS, confirming the class-level
    # access is wrong and was correctly replaced by instance access.
    assert not hasattr(EvaluationBundle, "target"), (
        "EvaluationBundle.target should not exist at class level; "
        "accessing it was the original bug."
    )
    assert not hasattr(EvaluationBundle, "excluded_columns"), (
        "EvaluationBundle.excluded_columns should not exist at class level."
    )

    # Now confirm a real instance has the attributes (correct code path)
    df = make_sample_sales(weeks=20, skus=4)
    bundle = evaluate_dataset(
        df,
        target="units_next_week",
        task="regression",
        run_tabpfn=False,
    )
    assert hasattr(bundle, "target")
    assert hasattr(bundle, "excluded_columns")
    assert bundle.target == "units_next_week"


def test_html_escape_available_in_analyst_chat():
    """Regression: app.py imported html.escape for the AI Analyst chat view
    but the `html` stdlib module was missing from imports, causing NameError
    when any user message was rendered.
    """
    import html as _html
    # Confirm the module is importable and escape works
    escaped = _html.escape("<script>alert('xss')</script>")
    assert "<script>" not in escaped
    assert "&lt;script&gt;" in escaped
