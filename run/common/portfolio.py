from typing import Sequence

import numpy as np


def softmax_np(x: Sequence[float], axis: int = -1) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return arr.astype(np.float32)
    arr = arr - np.max(arr, axis=axis, keepdims=True)
    exp_arr = np.exp(arr)
    return (exp_arr / (np.sum(exp_arr, axis=axis, keepdims=True) + 1e-12)).astype(np.float32)


def stock_weights_to_env_action(
    stock_weights: Sequence[float],
    *,
    normalize_if_overweight: bool = True,
    invest_all_when_positive: bool = False,
    empty_cash: bool = True,
) -> np.ndarray:
    """Convert stock-only weights to StockTradingEnv action: [cash, stock1..stockN]."""
    weights = np.asarray(stock_weights, dtype=np.float32).reshape(-1)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, 0.0, None)
    stock_sum = float(weights.sum())

    if stock_sum <= 1e-12:
        cash = 1.0 if empty_cash else 0.0
        return np.concatenate([[cash], np.zeros_like(weights)], axis=0).astype(np.float32)

    if invest_all_when_positive:
        weights = weights / (stock_sum + 1e-12)
        cash = 0.0
    else:
        if normalize_if_overweight and stock_sum > 1.0:
            weights = weights / (stock_sum + 1e-12)
            stock_sum = 1.0
        cash = max(0.0, 1.0 - stock_sum)

    action = np.concatenate([[cash], weights], axis=0).astype(np.float32)
    total = float(action.sum())
    if total <= 1e-12:
        action[:] = 0.0
        if empty_cash:
            action[0] = 1.0
    else:
        action /= total
    return action.astype(np.float32)
