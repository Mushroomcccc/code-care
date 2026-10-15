import csv
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


MONTHLY_SECTOR_COLUMNS = [
    "month",
    "start_date",
    "end_date",
    "sector",
    "monthly_return",
    "monthly_sharpe",
    "invested_days",
    "avg_weight",
]


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_csv_rows(path: str, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> str:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def save_amount_curve_csv(path: str, amount: Sequence[float], *, normalize: bool = False) -> str:
    amount_arr = np.asarray(amount, dtype=np.float64).reshape(-1)
    if normalize and amount_arr.size > 0 and abs(float(amount_arr[0])) > 1e-12:
        amount_arr = amount_arr / float(amount_arr[0])
    rows = [{"amount": float(x)} for x in amount_arr.tolist()]
    return write_csv_rows(path, fieldnames=["amount"], rows=rows)


def monthly_sector_metrics_to_frame(monthly_sector_metrics: Optional[List[dict]]):
    # Pandas is needed only by callers that explicitly request a DataFrame. Keep
    # lightweight training runners independent of this optional dependency.
    import pandas as pd

    rows = []
    for month_payload in monthly_sector_metrics or []:
        if not isinstance(month_payload, dict):
            continue

        month = month_payload.get("month")
        start_date = month_payload.get("start_date")
        end_date = month_payload.get("end_date")
        sector_metrics = month_payload.get("sector_metrics", {})
        if not isinstance(sector_metrics, dict):
            continue

        for sector_name, metrics in sector_metrics.items():
            metrics = metrics if isinstance(metrics, dict) else {}
            rows.append(
                {
                    "month": month,
                    "start_date": start_date,
                    "end_date": end_date,
                    "sector": sector_name,
                    "monthly_return": metrics.get("monthly_return"),
                    "monthly_sharpe": metrics.get("monthly_sharpe"),
                    "invested_days": metrics.get("invested_days"),
                    "avg_weight": metrics.get("avg_weight"),
                }
            )

    return pd.DataFrame(rows, columns=MONTHLY_SECTOR_COLUMNS)


def save_monthly_sector_metrics_csv(
    monthly_sector_metrics: Optional[List[dict]],
    out_path: str,
) -> str:
    ensure_parent_dir(out_path)
    # monthly_sector_metrics_to_frame(monthly_sector_metrics).to_csv(out_path, index=False)
    return out_path
