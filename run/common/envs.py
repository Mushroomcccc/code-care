import os
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

from env.stock_env import StockTradingEnv


@contextmanager
def temporary_cwd(path: Optional[str]) -> Iterator[None]:
    """Temporarily pin cwd for envs that read data through relative paths."""
    if not path:
        yield
        return

    cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


def make_stock_env(
    beg: int,
    end: int,
    market: str,
    sector_num: int = 3,
    *,
    project_root: Optional[str] = None,
    env_name: Optional[str] = None,
    gamma: Optional[float] = None,
    risk_window: Optional[int] = None,
    vol_penalty_coef: Optional[float] = None,
    drawdown_penalty_coef: Optional[float] = None,
    context_model: Optional[str] = None,
) -> StockTradingEnv:
    kwargs = {
        "beg_idx": beg,
        "end_idx": end,
        "market": market,
        "sector_num": sector_num,
    }
    optional_kwargs = {
        "env_name": env_name,
        "gamma": gamma,
        "risk_window": risk_window,
        "vol_penalty_coef": vol_penalty_coef,
        "drawdown_penalty_coef": drawdown_penalty_coef,
        "context_model": context_model,
    }
    kwargs.update({key: value for key, value in optional_kwargs.items() if value is not None})

    with temporary_cwd(project_root):
        return StockTradingEnv(**kwargs)


def split_622(total_len: int) -> Tuple[int, int, int, int, int, int]:
    """Return train/val/test index windows using the shared 6:2:2 split."""
    total_len = int(total_len)
    if total_len < 10:
        raise ValueError(f"Dataset too short for 6:2:2 split: total_len={total_len}")

    train_beg = 0
    train_end = max(2, int(total_len * 0.6))
    val_beg = train_end
    val_end = max(val_beg + 1, int(total_len * 0.8))
    val_end = min(val_end, total_len - 1)
    test_beg = val_end
    test_end = total_len

    if not (0 < train_end < val_end < test_end):
        raise ValueError(
            f"Invalid split: train=[{train_beg},{train_end}) "
            f"val=[{val_beg},{val_end}) test=[{test_beg},{test_end})"
        )
    return train_beg, train_end, val_beg, val_end, test_beg, test_end
