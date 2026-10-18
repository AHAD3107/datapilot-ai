"""Deterministic synthetic datasets for testing and safe, repeatable audits."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample_sales(seed: int = 27, weeks: int = 52, skus: int = 8) -> pd.DataFrame:
    """Original sample retail sales dataset."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-06")
    sku_ids = [f"ITEM-{i:03d}" for i in range(1, skus + 1)]
    base_demand = rng.integers(8, 40, size=skus)
    prices = rng.uniform(80, 900, size=skus).round(2)
    rows = []
    history: dict[str, list[int]] = {sku: [] for sku in sku_ids}
    for week in range(weeks):
        date = start + pd.DateOffset(weeks=week)
        seasonal = 1 + 0.18 * np.sin(2 * np.pi * week / 52)
        for idx, sku in enumerate(sku_ids):
            promotion = bool(rng.random() < (0.25 if week % 4 else 0.4))
            stock = int(rng.integers(5, 90))
            noise = rng.normal(0, 4)
            last = history[sku][-1] if history[sku] else int(base_demand[idx])
            avg4 = float(np.mean(history[sku][-4:])) if history[sku] else float(base_demand[idx])
            demand = max(0, int(round((base_demand[idx] * seasonal) + (8 if promotion else 0) + (last * 0.18) + noise)))
            sold = min(demand, stock)
            rows.append({
                "week_start": date.strftime("%Y-%m-%d"),
                "sku": sku,
                "price": float(prices[idx]),
                "promotion": "yes" if promotion else "no",
                "stock_available": stock,
                "units_last_week": int(last),
                "units_4wk_avg": round(avg4, 2),
                "units_next_week": sold,
            })
            history[sku].append(sold)
    return pd.DataFrame(rows)


def make_sales_regression_fixture(seed: int = 42, n_rows: int = 120) -> pd.DataFrame:
    """Fixture reproducing clean sales regression dataset."""
    rng = np.random.default_rng(seed)
    base_date = pd.Timestamp("2024-01-01")
    rows = []
    channels = ["online", "retail", "wholesale"]
    regions = ["north", "south", "east", "west"]
    for i in range(n_rows):
        dt = base_date + pd.DateOffset(days=int(i))
        ad_spend = float(rng.uniform(100, 5000))
        price = float(rng.uniform(20, 200))
        discount = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
        channel = rng.choice(channels)
        region = rng.choice(regions)
        sales = float(500 + 1.8 * ad_spend - 2.2 * price - 100 * discount + rng.normal(0, 50))
        rows.append({
            "date": dt.strftime("%Y-%m-%d"),
            "ad_spend": round(ad_spend, 2),
            "price": round(price, 2),
            "discount": discount,
            "channel": channel,
            "region": region,
            "units_sold": max(0.0, round(sales, 2)),
        })
    return pd.DataFrame(rows)


def make_customer_churn_fixture(seed: int = 101, n_rows: int = 240) -> pd.DataFrame:
    """Fixture reproducing customer churn classification dataset."""
    rng = np.random.default_rng(seed)
    rows = []
    contracts = ["month-to-month", "one-year", "two-year"]
    for i in range(n_rows):
        tenure = int(rng.integers(1, 72))
        monthly_charges = float(rng.uniform(20, 120))
        contract = rng.choice(contracts)
        support_calls = int(rng.integers(0, 6))
        churn_prob = 0.2
        if contract == "month-to-month":
            churn_prob += 0.25
        if support_calls >= 3:
            churn_prob += 0.3
        if tenure > 36:
            churn_prob -= 0.2
        churn = "yes" if rng.random() < max(0.05, min(0.95, churn_prob)) else "no"
        rows.append({
            "customer_id": f"CUST-{i:04d}",
            "tenure_months": tenure,
            "monthly_charges": round(monthly_charges, 2),
            "contract_type": contract,
            "support_calls": support_calls,
            "churn": churn,
        })
    return pd.DataFrame(rows)


def make_dirty_data_fixture(seed: int = 202, n_rows: int = 103) -> pd.DataFrame:
    """Fixture reproducing dirty data with missing values, missing targets, duplicates, constants."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        val_a = float(rng.normal(50, 15)) if rng.random() > 0.15 else np.nan
        category = rng.choice(["Alpha", "Beta", "Gamma"]) if rng.random() > 0.10 else None
        target_val = float(rng.uniform(10, 100)) if rng.random() > 0.08 else np.nan
        rows.append({
            "feature_numeric": val_a,
            "feature_category": category,
            "constant_col": "FIXED_CONSTANT",
            "all_nan_feature": np.nan,
            "target": target_val,
        })
    df = pd.DataFrame(rows)
    # Add exact duplicates
    if len(df) > 5:
        df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    return df


def make_target_leakage_fixture(seed: int = 303, n_rows: int = 160) -> pd.DataFrame:
    """Fixture reproducing the target leakage challenge dataset."""
    rng = np.random.default_rng(seed)
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    for i in range(n_rows):
        signup_dt = base_date + pd.DateOffset(days=int(i))
        usage_hours = float(rng.uniform(2, 80))
        n_logins = int(rng.integers(1, 30))
        plan = rng.choice(["starter", "pro", "enterprise"])
        renew_prob = 0.3 + 0.005 * usage_hours + 0.01 * n_logins
        will_renew = 1 if rng.random() < min(0.9, renew_prob) else 0
        # Target copy leakage feature
        renewal_decision_recorded = will_renew
        # Suspicious name feature
        post_decision_notes = f"Decision {will_renew} processed" if rng.random() > 0.5 else "Standard"
        rows.append({
            "signup_date": signup_dt.strftime("%Y-%m-%d"),
            "usage_hours": round(usage_hours, 2),
            "logins_count": n_logins,
            "plan_type": plan,
            "renewal_decision_recorded": renewal_decision_recorded,
            "post_decision_notes": post_decision_notes,
            "will_renew": will_renew,
        })
    return pd.DataFrame(rows)


def make_small_edge_case_fixture(n_rows: int = 10) -> pd.DataFrame:
    """Fixture reproducing a very small dataset for safeguard testing."""
    return pd.DataFrame({
        "id": range(n_rows),
        "score": [1.5 * i for i in range(n_rows)],
        "target": [i % 2 for i in range(n_rows)],
    })
