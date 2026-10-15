import csv
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.core as _np_core
import numpy.random as rd

# Keep compatibility with old pickle files that reference numpy._core.
sys.modules.setdefault('numpy._core', _np_core)


def _normalize_sector_names(sector_names: Optional[Any], num_sectors: int) -> List[str]:
    names: List[str] = []
    if sector_names is not None:
        try:
            raw = np.asarray(sector_names, dtype=object).reshape(-1)
            names = [str(x).strip() for x in raw.tolist() if str(x).strip()]
        except Exception:
            names = []
    if len(names) < num_sectors:
        names.extend([f"Sector-{k}" for k in range(len(names), num_sectors)])
    return names[:num_sectors]


def _build_default_sector_ids(num_stocks: int, num_sectors: int) -> np.ndarray:
    if num_stocks <= 0:
        raise ValueError(f"num_stocks must be positive, got {num_stocks}")
    if num_sectors <= 0:
        raise ValueError(f"num_sectors must be positive, got {num_sectors}")

    sector_ids = np.zeros(num_stocks, dtype=np.int64)
    width = float(num_stocks) / float(num_sectors)
    for i in range(num_stocks):
        sector_ids[i] = min(int(i / width), num_sectors - 1)
    return sector_ids


def _build_sector_to_indices(stock_to_sector: np.ndarray, num_sectors: int) -> List[np.ndarray]:
    return [np.where(stock_to_sector == k)[0] for k in range(num_sectors)]


def _resolve_market_context_csv(
    market: str,
    data_dir: Optional[str] = None,
    model: Optional[str] = None,
) -> Path:
    base_dir = Path(data_dir).expanduser().resolve() if data_dir else Path(__file__).resolve().parents[1] / "data"
    preferred_paths: List[Path] = []

    if model:
        preferred_paths.append(base_dir / f"{market}_context_{model}.csv")
    preferred_paths.append(base_dir / f"{market}_context.csv")
    preferred_paths.append(base_dir / f"{market}_stock_with_news.csv")

    for csv_path in preferred_paths:
        if csv_path.exists():
            return csv_path

    variant_paths = sorted(base_dir.glob(f"{market}_context_*.csv"))
    if len(variant_paths) == 1:
        return variant_paths[0]

    searched = ", ".join(str(path.name) for path in preferred_paths)
    if variant_paths:
        available = ", ".join(path.name for path in variant_paths)
        raise FileNotFoundError(
            f"market context file not found (searched: {searched}; available variants: {available})"
        )
    raise FileNotFoundError(f"market context file not found (searched: {searched})")


@lru_cache(maxsize=16)
def _load_market_dates(
    market: str,
    data_dir: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, ...]:
    market = str(market).strip()
    if not market:
        raise ValueError("market must be a non-empty string")

    csv_path = _resolve_market_context_csv(market=market, data_dir=data_dir, model=model)

    dates: List[str] = []
    seen_dates = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        date_field = "date" if "date" in fieldnames else "Date" if "Date" in fieldnames else None
        if date_field is None:
            raise ValueError(f"context csv {csv_path} is missing a 'date' or 'Date' column")

        for row in reader:
            date = str(row.get(date_field, "")).strip()
            if date and date not in seen_dates:
                dates.append(date)
                seen_dates.add(date)

    if not dates:
        raise ValueError(f"context csv {csv_path} does not contain any usable dates")
    return tuple(dates)


def _compute_period_sharpe(daily_returns: np.ndarray) -> float:
    if daily_returns.size < 2:
        return 0.0

    std = float(np.std(daily_returns))
    if std <= 1e-12:
        return 0.0
    return float(math.sqrt(float(daily_returns.size)) * float(np.mean(daily_returns)) / std)

class StockTradingEnv:
    def __init__(
        self,
        initial_amount=1e6,
        buy_cost_pct=1e-4, #org:1e-3
        sell_cost_pct=1e-4,#org:1e-3
        gamma=0.99,
        risk_window: int = 20,
        vol_penalty_coef: float = 0.05,
        drawdown_penalty_coef: float = 0.01,
        beg_idx=0,
        end_idx=1250,
        market='US',
        sector_num: Optional[int] = None,
        stock_to_sector: Optional[np.ndarray] = None,
        env_name:str = '',
        context_model: Optional[str] = None,
    ):

        # --- Data loading ---
        #US ChA LSE JAN HKEX ASX
        self.market = market 
        self.name = env_name
        self.context_model = str(context_model).strip() if context_model is not None else None
        self.data_dir = str(Path(__file__).resolve().parents[1] / "data")
        self.npz_pwd = f'./data/{market}_{sector_num * 6}.npz'
        self.sector_names = np.load(f'./data/{market}_snames{sector_num}.npz', allow_pickle=True)["sector_name"]

        self.close_ary, self.tech_ary, self.low_ary = self.load_data_from_disk()
        # Retain read-only references to the complete history.  Split environments
        # still trade strictly inside [beg_idx, end_idx), while causal feature
        # builders can use observations before beg_idx instead of restarting every
        # rolling window at a split boundary.
        self.full_close_ary = self.close_ary
        self.full_tech_ary = self.tech_ary
        self.full_low_ary = self.low_ary
        self.full_data_len = int(self.close_ary.shape[0])
        self.beg_idx = min(max(int(beg_idx), 0), self.full_data_len)
        requested_end_idx = self.full_data_len if end_idx is None else int(end_idx)
        self.end_idx = min(max(requested_end_idx, self.beg_idx), self.full_data_len)
        self.stocks_num = self.close_ary.shape[1]
        self.close_ary = self.close_ary[self.beg_idx:self.end_idx]
        self.close_ary = self.close_ary
        self.tech_ary = self.tech_ary[self.beg_idx:self.end_idx]
        self.low_ary = self.low_ary[self.beg_idx:self.end_idx]
        self.low_ary = self.low_ary

        inferred_sector_num = int(len(np.asarray(self.sector_names).reshape(-1)))
        self.sector_num = int(sector_num) if sector_num is not None else inferred_sector_num
        if self.sector_num <= 0:
            raise ValueError(f"sector_num must be positive, got {self.sector_num}")
        self.sector_names = _normalize_sector_names(self.sector_names, self.sector_num)

        # --- Basic parameters ---
        self.buy_cost_rate = 1 + buy_cost_pct
        self.sell_cost_rate = 1 - sell_cost_pct
        # LSE JAN HKEX US
        # if market == 'HKEX':
        #     initial_amount = initial_amount / 10
        # if market == 'LSE' or market == 'JAN':
        #     initial_amount = initial_amount * 10
        self.initial_amount = initial_amount
        self.gamma = gamma
        self.risk_window = max(int(risk_window), 1)
        self.vol_penalty_coef = float(vol_penalty_coef)
        self.drawdown_penalty_coef = float(drawdown_penalty_coef)

        # reset()
        self.day = None
        self.rewards = None
        self.raw_returns = None
        self.total_asset = None
        self.cumulative_returns = 0
        self.if_random_reset = True #True
        self.peak_asset = None
        self.max_drawdown = 0.0
        self.enable_monthly_sector_stats = False
        self.monthly_sector_stats_error = None
        self.market_dates: Tuple[str, ...] = tuple()
        self.slice_dates: Tuple[str, ...] = tuple()
        self.monthly_sector_metrics = None
        self.latest_monthly_sector_metrics = None
        self._active_month_key = None
        self._active_month_dates = None
        self._sector_monthly_daily_returns = None
        self._sector_monthly_weights = None

        try:
            market_dates = _load_market_dates(
                self.market,
                data_dir=self.data_dir,
                model=self.context_model,
            )
            if len(market_dates) != self.full_data_len:
                raise ValueError(
                    f"market date count mismatch: {len(market_dates)} vs {self.full_data_len}"
                )
            self.market_dates = market_dates
            self.slice_dates = market_dates[self.beg_idx:self.end_idx]
            self.enable_monthly_sector_stats = len(self.slice_dates) == len(self.close_ary)
            if not self.enable_monthly_sector_stats:
                self.monthly_sector_stats_error = (
                    f"slice date count mismatch: {len(self.slice_dates)} vs {len(self.close_ary)}"
                )
        except (FileNotFoundError, ValueError) as exc:
            self.monthly_sector_stats_error = str(exc)

        # Number of stocks.
        self.shares_num = self.close_ary.shape[1]
        amount_dim = 1

        if self.tech_ary.shape[1] % self.shares_num != 0:
            raise ValueError(
                f"tech_ary second dim {self.tech_ary.shape[1]} is not divisible by shares_num {self.shares_num}"
            )
        self.tech_dim_per_stock = int(self.tech_ary.shape[1] // self.shares_num)

        if stock_to_sector is None:
            stock_to_sector = _build_default_sector_ids(self.shares_num, self.sector_num)
        else:
            stock_to_sector = np.asarray(stock_to_sector, dtype=np.int64).reshape(-1)
            if stock_to_sector.shape[0] != self.shares_num:
                raise ValueError(
                    f"stock_to_sector length mismatch: {stock_to_sector.shape[0]} vs {self.shares_num}"
                )
            if np.any((stock_to_sector < 0) | (stock_to_sector >= self.sector_num)):
                raise ValueError(
                    f"stock_to_sector values must be in [0, {self.sector_num - 1}]"
                )

        self.stock_to_sector = stock_to_sector.astype(np.int64, copy=False)
        self.sector_to_indices = _build_sector_to_indices(self.stock_to_sector, self.sector_num)
        self.max_sector_stocks = max((int(idx.size) for idx in self.sector_to_indices), default=0)
        self.maddpg_state_dim = (
            self.max_sector_stocks * (self.tech_dim_per_stock + 2) + 2
        )

        # environment information
        self.env_name = 'StockTradingEnv-v1'

        # --- State dimension (modified) ---
        # Fix duplicate counting in the original definition and make it clearer.
        # State: [cash, share holdings, current-day close prices, technical indicators]
        self.state_dim = amount_dim + self.shares_num + self.close_ary.shape[1] + self.tech_ary.shape[1] 

        # --- Action dimension (modified) ---
        # Action: [stock 1 weight, ..., stock N weight, cash weight]
        self.action_dim = self.shares_num + 1  # <--- MODIFIED

        self.max_step = len(self.close_ary)

    def reset(self):
        self.day = 0
        if self.if_random_reset:
            self.amount = self.initial_amount * rd.uniform(0.8, 1.2)
            self.shares = (np.abs(rd.randn(self.shares_num).clip(-2, +2)) * 2 ** 6).astype(int)
           # self.shares = np.zeros(self.shares_num, dtype=np.float32)

        else:
            self.amount = self.initial_amount
            self.shares = np.zeros(self.shares_num, dtype=np.float32)

        self.rewards = list()
        self.raw_returns = list()
        # Compute total_asset during reset based on day-0 prices.
        self.total_asset = (self.close_ary[self.day] * self.shares).sum() + self.amount
        self.init_asset = self.total_asset
        self.peak_asset = float(self.total_asset)
        self.max_drawdown = 0.0
        self._reset_monthly_sector_tracking()
        return self.get_state()

    def _reset_monthly_sector_tracking(self) -> None:
        self.monthly_sector_metrics = []
        self.latest_monthly_sector_metrics = None
        self._activate_month_bucket(month_key=None)

    def _activate_month_bucket(self, month_key: Optional[str]) -> None:
        self._active_month_key = month_key
        self._active_month_dates = []
        self._sector_monthly_daily_returns = [[] for _ in range(self.sector_num)]
        self._sector_monthly_weights = [[] for _ in range(self.sector_num)]

    def _finalize_active_month_metrics(self) -> Optional[Dict[str, Any]]:
        if not self._active_month_key or not self._active_month_dates:
            return None

        sector_metrics: Dict[str, Dict[str, Any]] = {}
        for sector_id, sector_name in enumerate(self.sector_names):
            daily_returns = np.asarray(self._sector_monthly_daily_returns[sector_id], dtype=np.float64)
            monthly_return = (
                float(np.prod(1.0 + daily_returns) - 1.0)
                if daily_returns.size > 0
                else None
            )
            monthly_sharpe = (
                float(_compute_period_sharpe(daily_returns))
                if daily_returns.size > 0
                else None
            )
            avg_weight = float(np.mean(self._sector_monthly_weights[sector_id])) if self._sector_monthly_weights[sector_id] else 0.0

            sector_metrics[sector_name] = {
                "monthly_return": monthly_return,
                "monthly_sharpe": monthly_sharpe,
                "invested_days": int(daily_returns.size),
                "avg_weight": avg_weight,
            }

        monthly_metrics = {
            "month": self._active_month_key,
            "start_date": self._active_month_dates[0],
            "end_date": self._active_month_dates[-1],
            "sector_metrics": sector_metrics,
        }
        self.monthly_sector_metrics.append(monthly_metrics)
        return monthly_metrics

    def _update_monthly_sector_metrics(
        self,
        start_prices: np.ndarray,
        end_prices: np.ndarray,
    ) -> None:
        self.latest_monthly_sector_metrics = None
        if not self.enable_monthly_sector_stats or not self.slice_dates:
            return

        return_day_index = min(self.day, len(self.slice_dates) - 1)
        return_date = self.slice_dates[return_day_index]
        month_key = return_date[:7]

        if self._active_month_key is None:
            self._activate_month_bucket(month_key=month_key)
        elif month_key != self._active_month_key:
            self.latest_monthly_sector_metrics = self._finalize_active_month_metrics()
            self._activate_month_bucket(month_key=month_key)

        start_stock_values = np.asarray(start_prices, dtype=np.float64) * np.asarray(self.shares, dtype=np.float64)
        end_stock_values = np.asarray(end_prices, dtype=np.float64) * np.asarray(self.shares, dtype=np.float64)
        portfolio_start_value = float(start_stock_values.sum() + self.amount)

        self._active_month_dates.append(return_date)
        for sector_id, idx in enumerate(self.sector_to_indices):
            sector_start_value = float(start_stock_values[idx].sum())
            sector_weight = sector_start_value / (portfolio_start_value + 1e-12)
            self._sector_monthly_weights[sector_id].append(float(sector_weight))
            if sector_start_value <= 1e-12:
                continue

            sector_end_value = float(end_stock_values[idx].sum())
            sector_daily_return = sector_end_value / (sector_start_value + 1e-12) - 1.0
            self._sector_monthly_daily_returns[sector_id].append(float(sector_daily_return))

        if self.day == self.max_step - 1:
            self.latest_monthly_sector_metrics = self._finalize_active_month_metrics()

    def _portfolio_weights_full(self, prices: np.ndarray) -> np.ndarray:
        """Compute w_full = [cash, stocks...] from current holdings."""
        stock_values = prices * self.shares
        total = float(stock_values.sum() + self.amount)
        if total <= 1e-12:
            w_cash = 1.0
            w_stock = np.zeros(self.shares_num, dtype=np.float64)
        else:
            w_cash = float(self.amount / total)
            w_stock = stock_values / total
        w_full = np.concatenate([[w_cash], w_stock], axis=0)
        w_full = np.clip(w_full, 0.0, None)
        return w_full / (w_full.sum() + 1e-12)

    def get_state(self):
        # Normalize the state; powers-of-two scaling is more efficient to compute.
        prices = self.close_ary[self.day]
        w_full = self._portfolio_weights_full(prices)
        # ta = np.array([self.total_asset / self.init_asset])
        # Use the same historical base across train/validation/test.  Normalising
        # by the first row of each split creates an avoidable distribution shift
        # (validation/test always restart at one) even though the global base is
        # already known at every later decision date.
        state = np.hstack((self.close_ary[self.day] / self.full_close_ary[0],
                           self.tech_ary[self.day],
                           w_full))
        return state

    def get_maddpg_state(self) -> List[np.ndarray]:
        """
        Return one local stock state per sector for low-level MADDPG agents.
        Each vector has the format:
        [sector_close, sector_tech, sector_stock_weights, sector_weight, cash_weight]
        The first three segments are zero-padded to max_sector_stocks for a fixed length.
        """
        prices = np.asarray(self.close_ary[self.day])
        # base_prices = np.maximum(np.asarray(self.close_ary[0], dtype=np.float64), 1e-12)
        # base_prices = np.max(np.array(self.close_ary, dtype=np.float64), axis=0)
        base_prices = self.close_ary[0]
        tech_day = np.asarray(self.tech_ary[self.day], dtype=np.float64).reshape(
            self.shares_num, self.tech_dim_per_stock
        )
        w_full = self._portfolio_weights_full(prices)
        stock_weights = np.asarray(w_full[1:], dtype=np.float64)
        cash_weight = float(w_full[0])

        maddpg_state: List[np.ndarray] = []
        for idx in self.sector_to_indices:
            sector_prices = prices[idx] / base_prices[idx]
            sector_tech = tech_day[idx].reshape(-1)
            sector_weights = stock_weights[idx]
            sector_weight = float(sector_weights.sum())

            pad_n = self.max_sector_stocks - int(idx.size)
            if pad_n > 0:
                sector_prices = np.pad(sector_prices, (0, pad_n), mode="constant")
                sector_tech = np.pad(
                    sector_tech,
                    (0, pad_n * self.tech_dim_per_stock),
                    mode="constant",
                )
                sector_weights = np.pad(sector_weights, (0, pad_n), mode="constant")

            sector_state = np.concatenate(
                [
                    sector_prices,
                    sector_tech,
                    sector_weights,
                    np.asarray([sector_weight, cash_weight], dtype=np.float64),
                ],
                axis=0,
            ).astype(np.float32)
            maddpg_state.append(sector_state)

        return maddpg_state

    def step(self, action):
        """
        Execute one trading step.
        action (np.array): shape (action_dim,) = (shares_num + 1,)
                           represents [stock 1...N logits, cash logit]
        """
        # 1. Save total assets at t-1 for reward calculation.
        previous_total_asset = self.total_asset

        # 2. Current stock prices.
        current_prices = self.close_ary[self.day]  # Get prices at time t.

        # 3. Compute total assets at time t before trading.
        current_portfolio_value = (current_prices * self.shares).sum() + self.amount

        # Record pre-trade holdings for turnover calculation.
        prev_shares = self.shares.copy()

        # 4. Convert action logits to target weights.
        if np.sum(np.abs(action)) == 0:
            target_weights_full = np.zeros(self.action_dim)
            target_weights_full[0] = 1.0  # 100% cash.
        else:
            target_weights_full = action / np.sum(action)

        target_stock_weights = target_weights_full[1:]
        target_cash_weight = target_weights_full[0]

        # 5. Compute target market values.
        target_stock_values = current_portfolio_value * target_stock_weights

        # 6. Compute current stock market values.
        current_stock_values = current_prices * self.shares

        # 7. Compute market value differences (target - current).
        delta_stock_values = target_stock_values - current_stock_values

        # 8. Execute trades, selling before buying to avoid cash shortfalls.

        # --- 8.1 Handle sells (delta_values < 0) ---
        sell_indices = np.where(delta_stock_values < 0)[0]
        for index in sell_indices:
            value_to_sell = -delta_stock_values[index]

            if current_prices[index] > 0:
                shares_to_sell = (value_to_sell / current_prices[index]).astype(int)
            else:
                shares_to_sell = 0

            shares_to_sell = min(shares_to_sell, self.shares[index])

            self.amount += current_prices[index] * shares_to_sell * self.sell_cost_rate
            self.shares[index] -= shares_to_sell

        # --- 8.2 Handle buys (delta_values > 0) ---
        buy_indices = np.where(delta_stock_values > 0)[0]
        sorted_buy_indices = sorted(buy_indices, key=lambda i: delta_stock_values[i], reverse=True)

        for index in sorted_buy_indices:
            value_to_buy = delta_stock_values[index]

            price_with_cost = current_prices[index] * self.buy_cost_rate
            if price_with_cost <= 1e-6:
                continue

            desired_shares = (value_to_buy / price_with_cost).astype(int)
            affordable_shares = (self.amount / price_with_cost).astype(int)

            shares_to_buy = min(desired_shares, affordable_shares)

            cost = shares_to_buy * price_with_cost
            if self.amount < 0:
                print(self.amount)
            self.amount -= cost
            self.shares[index] += shares_to_buy

        # 9. Compute turnover for this period based on prices at time t.
        # Traded value = |delta shares| * price, counting both buys and sells.
        trade_values = np.abs(self.shares - prev_shares) * current_prices
        turnover = trade_values.sum() / (previous_total_asset + 1e-8)

        if self.amount < 0:
            print(f'| 警告：Day {self.day} 现金变为负数 {self.amount}')
            self.amount = 0


        # Use the close price at t + 1 to compute post-trade total assets at t + 1.
        self.day += 1
        next_prices = self.close_ary[min(self.day, self.close_ary.shape[0] - 1)]
        new_total_asset = (next_prices * self.shares).sum() + self.amount

        # Raw return component: log return.
        raw_return = math.log(new_total_asset / previous_total_asset)
        self.raw_returns.append(float(raw_return))
        self._update_monthly_sector_metrics(start_prices=current_prices, end_prices=next_prices)

        recent_raw_returns = self.raw_returns[-self.risk_window :]
        rolling_vol = (
            float(np.std(np.asarray(recent_raw_returns, dtype=np.float64)))
            if len(recent_raw_returns) >= 2
            else 0.0
        )

        self.peak_asset = max(float(self.peak_asset), float(new_total_asset))
        drawdown = max(0.0, 1.0 - float(new_total_asset) / (float(self.peak_asset) + 1e-12))
        self.max_drawdown = max(float(self.max_drawdown), float(drawdown))

        # Debug info still keeps turnover, while the training objective uses risk-adjusted returns.
        penalty = 0.001 * turnover # org 0.001
        vol_penalty = self.vol_penalty_coef * rolling_vol
        drawdown_penalty = self.drawdown_penalty_coef * drawdown
        reward = raw_return #- vol_penalty - drawdown_penalty

        self.rewards.append(reward)
        self.total_asset = new_total_asset

        # Keep the Sharpe ratio part unchanged for high-level h_reward.
        returns = np.array(self.raw_returns[-10:])
        sharpe_ratio = returns.mean() / (returns.std() + 1e-6)
        sharpe_ratio = 0 if len(self.raw_returns) < 5 else sharpe_ratio
        h_reward = raw_return - vol_penalty - drawdown_penalty #sharpe_ratio

        # Final low-level reward = return - volatility penalty - drawdown penalty.

        done = self.day == self.max_step - 1
        
        self.cumulative_returns = self.total_asset / self.init_asset
        G = 0
        if done:
            for r in reversed(self.rewards):
                G = r + G # self.gamma * G
        info = {
            "cumret": self.cumulative_returns if done else None,
            "cumred": G if done else None,
            "h_reward": h_reward,
            "raw_return": float(raw_return),
            "rolling_vol": float(rolling_vol),
            "vol_penalty": float(vol_penalty),
            "drawdown": float(drawdown),
            "max_drawdown": float(self.max_drawdown),
            "drawdown_penalty": float(drawdown_penalty),
            "risk_adjusted_reward": float(reward),
            "turnover": turnover,          # <--- NEW: useful for debugging turnover.
            "turnover_penalty": penalty,   # <--- NEW: penalty value for the current step.
            "latest_monthly_sector_metrics": self.latest_monthly_sector_metrics,
            "monthly_sector_metrics": self.monthly_sector_metrics if done else None,
            "monthly_sector_stats_enabled": bool(self.enable_monthly_sector_stats),
            "monthly_sector_stats_error": self.monthly_sector_stats_error,
            "maddpg_state": self.get_maddpg_state(),
        }
        state = self.get_state()

        return state, reward, done, info

    # <--- MODIFIED: added dependency checks for os and pd. ---
    def load_data_from_disk(self, tech_id_list=None):
        tech_id_list = [
            "macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma",
        ] if tech_id_list is None else tech_id_list

        ary_dict = np.load(self.npz_pwd, allow_pickle=True)
        close_ary = ary_dict['close_ary']
        tech_ary = ary_dict['tech_ary']
        low_ary = ary_dict["low_ary"]

        return close_ary, tech_ary, low_ary

def kwargs_filter(func, kwargs: dict):
    """
    Filter the variable in env func.
    :param func: the function for creating an env.
    :param kwargs: args for the env.
    :return: filtered args.
    """
    import inspect

    sign = inspect.signature(func).parameters.values()
    sign = {val.name for val in sign}

    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def build_env(env_func=None, env_args=None):
    env = env_func(**kwargs_filter(env_func.__init__, env_args.copy()))
    return env
