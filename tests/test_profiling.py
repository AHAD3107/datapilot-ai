import pandas as pd

from datapilot.profiling import infer_task, profile_dataframe
from datapilot.sample_data import make_sample_sales


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
