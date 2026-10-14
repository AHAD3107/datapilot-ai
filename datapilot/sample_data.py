"""Deterministic synthetic sales dataset for safe, repeatable demos."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample_sales(seed: int = 27, weeks: int = 52, skus: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-06")
    sku_ids = [f"ITEM-{i:03d}" for i in range(1, skus + 1)]
    base_demand = rng.integers(8, 40, size=skus)
    prices = rng.uniform(80, 900, size=skus).round(2)
    rows = []
    history: dict[str, list[int]] = {sku: [] for sku in sku_ids}
    for week in range(weeks):
        date = start + pd.Timedelta(weeks=week)
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
