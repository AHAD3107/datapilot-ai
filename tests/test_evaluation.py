import numpy as np
import pandas as pd
import pytest

from datapilot.evaluation import build_decision, evaluate_dataset
from datapilot.sample_data import make_sample_sales


def test_regression_evaluation_runs_baselines_without_tabpfn():
    df = make_sample_sales(seed=9, weeks=32, skus=4)
    bundle = evaluate_dataset(
        df, target="units_next_week", task="regression", time_col="week_start",
        excluded_columns=["sku"], run_tabpfn=False,
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
    bundle = evaluate_dataset(df, target="units_next_week", task="regression", time_col="week_start", run_tabpfn=False)
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
