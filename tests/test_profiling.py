import numpy as np
import pandas as pd
import pytest

from datapilot.profiling import _detect_corrupted_numeric, infer_task, profile_dataframe
from datapilot.sample_data import make_dirty_data_fixture, make_sample_sales


def test_sample_data_is_deterministic_and_has_expected_columns():
    first = make_sample_sales(seed=7, weeks=10, skus=3)
    second = make_sample_sales(seed=7, weeks=10, skus=3)
    pd.testing.assert_frame_equal(first, second)
    assert {"week_start", "sku", "units_next_week"}.issubset(first.columns)
    assert len(first) == 30


def test_profile_reports_missing_duplicates_and_suspicious_target_names():
    df = pd.DataFrame({
        "price": [2, 3, 3, 4],
        "future_result": [1, 0, 1, 1],
        "target": [1, 0, 1, 1],
        "note": [None, "a", "b", "c"],
    })
    df = pd.concat([df, df.iloc[[1]]], ignore_index=True)
    profile = profile_dataframe(df, target="target")
    assert profile["duplicate_rows"] == 1
    assert profile["missing_cells"] == 1
    assert "future_result" in profile["suspicious_columns"]
    assert profile["warnings"]


def test_task_inference():
    assert infer_task(pd.Series(["yes", "no", "yes"])) == "classification"
    assert infer_task(pd.Series([1.4, 2.3, 3.7, 4.5, 6.2, 8.1, 10.4, 12.8, 14.2, 17.4, 20.1, 25.9, 30.2])) == "regression"


def test_dirty_data_fixture_profiling():
    df = make_dirty_data_fixture(seed=42, n_rows=103)
    profile = profile_dataframe(df, target="target")
    assert profile["missing_cells"] > 0
    assert profile["duplicate_rows"] >= 1
    assert "FIXED_CONSTANT" in df["constant_col"].values
    assert "constant_col" in profile["constant_columns"]
    assert profile["target_summary"]["missing"] > 0


def test_empty_dataframe_profiling_raises():
    with pytest.raises(ValueError, match="no rows"):
        profile_dataframe(pd.DataFrame())


def test_corrupted_numeric_currency_and_percentages_detected():
    df = pd.DataFrame({
        "clean_num": [10.5, 20.0, 35.2, 40.1, 55.9] * 8,
        "currency_col": ["$1,200", "$450", "$300.50", "$2,100", "$800"] * 8,
        "pct_col": ["12%", "25%", "50%", "75%", "99%"] * 8,
        "unit_col": ["45 kg", "10 kg", "80 kg", "12 kg", "95 kg"] * 8,
        "normal_cat": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"] * 8,
        "target": [1, 0, 1, 0, 1] * 8,
    })
    profile = profile_dataframe(df, target="target")
    corrupted_cols = [c["column"] for c in profile["corrupted_numeric"]]
    assert "currency_col" in corrupted_cols
    assert "pct_col" in corrupted_cols
    assert "unit_col" in corrupted_cols
    assert "clean_num" not in corrupted_cols
    assert "normal_cat" not in corrupted_cols

    # Ensure warning is surfaced
    cn_warnings = [w for w in profile["warnings"] if "Corrupted numeric column" in w["title"]]
    assert len(cn_warnings) == 3


def test_corrupted_numeric_mixed_values_and_missing_handling():
    # Mixed strings with non-numeric N/A tokens
    series = pd.Series(["100", "250", "N/A", "400", "500", "650", None, "800", "950", "#REF!"])
    res = _detect_corrupted_numeric(series, "mixed_score")
    assert res is not None
    assert res["column"] == "mixed_score"
    assert res["clean_rate"] >= 70.0


def test_corrupted_numeric_small_sample_and_low_cardinality_safe():
    # Low-cardinality boolean-like text should not be flagged as corrupted numeric
    low_card = pd.Series(["yes", "no", "yes", "no", "yes"])
    assert _detect_corrupted_numeric(low_card, "flag") is None

    # Very small sample (< 5 rows)
    small_s = pd.Series(["$100", "$200", "$300"])
    assert _detect_corrupted_numeric(small_s, "price") is None


def test_profile_column_type_counts_and_schema():
    df = pd.DataFrame({
        "num_int": [1, 2, 3, 4, 5],
        "num_float": [1.1, 2.2, 3.3, 4.4, 5.5],
        "cat_str": ["apple", "banana", "cherry", "date", "elderberry"],
        "cat_bool": [True, False, True, True, False],
        "dt_col": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
    })
    profile = profile_dataframe(df)

    # Check schema presence
    assert "numeric_columns" in profile
    assert "categorical_columns" in profile
    assert "date_columns" in profile
    assert "memory_mb" in profile
    assert "suggested_exclusions" in profile

    # Check correct partitioning
    assert set(profile["numeric_columns"]) == {"num_int", "num_float"}
    assert set(profile["categorical_columns"]) == {"cat_str", "cat_bool"}
    assert set(profile["date_columns"]) == {"dt_col"}

    # Ensure datetime columns are not counted as categorical
    assert "dt_col" not in profile["categorical_columns"]
    assert "dt_col" not in profile["numeric_columns"]

    # Total count sum matches number of columns
    total_classified = len(profile["numeric_columns"]) + len(profile["categorical_columns"]) + len(profile["date_columns"])
    assert total_classified == profile["columns"] == 5
    assert profile["memory_bytes"] > 0
    assert profile["memory_mb"] >= 0.0
