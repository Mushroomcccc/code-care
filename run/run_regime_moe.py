from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

from agent.regime_moe_agent import (
    EWMAExpertReliability,
    REGIME_NAMES,
    RegimeMoEConfig,
    RegimeMoEPortfolioAgent,
    RegimeMoETransition,
    RegimePriorProvider,
    compute_market_features,
    compute_policy_market_features,
    compute_asset_features,
    compute_causal_expert_anchors,
    compute_specialization_signals,
    heuristic_regime_weights,
)
from env.stock_env import StockTradingEnv
from run.common.envs import make_stock_env, split_622
from run.common.io import save_amount_curve_csv, save_monthly_sector_metrics_csv, write_csv_rows
from run.common.seeding import set_seed


LOGGER = logging.getLogger("run_regime_moe")


class _SafeStreamHandler(logging.StreamHandler):
    def handleError(self, record: logging.LogRecord) -> None:
        exc_type, exc_value, _ = sys.exc_info()
        is_closed_stream = (
            exc_type is not None
            and issubclass(exc_type, ValueError)
            and "closed" in str(exc_value).lower()
        )
        is_broken_pipe = (
            exc_type is not None
            and issubclass(exc_type, OSError)
            and (issubclass(exc_type, BrokenPipeError) or getattr(exc_value, "errno", None) == 32)
        )
        if is_closed_stream or is_broken_pipe:
            logging.getLogger(record.name).removeHandler(self)
            return
        super().handleError(record)


@dataclass
class TrajectoryResult:
    transitions: List[RegimeMoETransition]
    amount: np.ndarray
    reward_sum: float
    discounted_reward: float
    cumret: float
    stats: Dict[str, Any]
    diagnostics: List[Dict[str, Any]]
    expert_actions: np.ndarray
    portfolio_actions: np.ndarray


def _setup_logger(market: str, method_name: str = "run_regime_moe") -> Tuple[logging.Logger, str, str]:
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    log_dir = os.path.join(SCRIPT_DIR, "logs", str(market).strip() or "unknown_market")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{method_name}_{timestamp}.log")

    logger = logging.getLogger(method_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter(fmt="%(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = _SafeStreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path, timestamp


def _log_run_config(logger: logging.Logger, cfg: argparse.Namespace) -> None:
    logger.info("[Config] ===== Run Parameters =====")
    for key in sorted(vars(cfg).keys()):
        logger.info("[Config] %-28s: %s", key, getattr(cfg, key))
    logger.info("[Config] ==========================")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_manifest(cfg: argparse.Namespace, *, status: str, extra: Optional[Dict[str, Any]] = None) -> str:
    out_dir = _get_results_out_dir(cfg.market, cfg)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"manifest_{cfg.seed}_{cfg.timestamp}.json")
    source_paths = [
        os.path.join(PROJECT_DIR, "run", "run_regime_moe.py"),
        os.path.join(PROJECT_DIR, "agent", "regime_moe_agent.py"),
        os.path.join(PROJECT_DIR, "env", "stock_env.py"),
        os.path.abspath(os.path.join(PROJECT_DIR, "..", "Method.tex")),
    ]
    prior_path = str(cfg.prior_cache).strip()
    if prior_path:
        source_paths.append(prior_path if os.path.isabs(prior_path) else os.path.join(PROJECT_DIR, prior_path))
    hashes = {os.path.relpath(p, PROJECT_DIR): _sha256(p) for p in source_paths if os.path.isfile(p)}
    payload: Dict[str, Any] = {
        "status": str(status),
        "market": str(cfg.market),
        "seed": int(cfg.seed),
        "timestamp": str(cfg.timestamp),
        "config": dict(vars(cfg)),
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "sha256": hashes,
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def _portfolio_metrics(amount: np.ndarray, annualization: int = 252) -> Dict[str, float]:
    values = np.asarray(amount, dtype=np.float64).reshape(-1)
    daily = values[1:] / values[:-1] - 1.0
    if daily.size < 2:
        return {key: float("nan") for key in ("ARR", "SR", "SOR", "CR", "MDD")}
    arr = (values[-1] / values[0] - 1.0) * float(annualization) / float(daily.size)
    std = float(np.std(daily, ddof=1))
    downside = float(np.sqrt(np.mean(np.minimum(daily, 0.0) ** 2)))
    drawdown = 1.0 - values / (np.maximum.accumulate(values) + 1e-12)
    mdd = float(np.max(drawdown))
    return {
        "ARR": float(arr),
        "SR": float(np.mean(daily) / (std + 1e-12) * np.sqrt(annualization)),
        "SOR": float(np.mean(daily) / (downside + 1e-12) * np.sqrt(annualization)),
        "CR": float(arr / (mdd + 1e-12)),
        "MDD": mdd,
    }


def _validation_selection_score(
    cumulative_return: float,
    metrics: Dict[str, float],
    selection_metric: str,
) -> float:
    """Validation-only checkpoint score; held-out test data is never consulted."""

    name = str(selection_metric)
    if name == "cumret":
        return float(cumulative_return)
    if name not in {"ARR", "SR", "SOR", "CR"}:
        raise ValueError(f"unsupported checkpoint selection metric: {name}")
    return float(metrics[name])


def _resolve_device(device: str) -> str:
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("[Device] Requested %s but CUDA is unavailable; using cpu.", device)
        return "cpu"
    return device


def make_env(beg: int, end: int, name: str, cfg: argparse.Namespace) -> StockTradingEnv:
    env = make_stock_env(
        beg,
        end,
        cfg.market,
        sector_num=cfg.num_sectors,
        project_root=PROJECT_DIR,
        env_name=name,
        gamma=cfg.gamma,
        risk_window=cfg.risk_window,
        vol_penalty_coef=cfg.vol_penalty_coef,
        drawdown_penalty_coef=cfg.drawdown_penalty_coef,
        context_model=cfg.context_model,
    )
    env.if_random_reset = bool(cfg.random_reset)
    return env


def _get_results_out_dir(market: str, cfg: argparse.Namespace) -> str:
    base = str(cfg.results_dir_name).strip() or "results_regime_moe"
    return os.path.join(PROJECT_DIR, base, market)


def _compute_discounted_return(rewards: Sequence[float], gamma: float) -> float:
    total = 0.0
    for reward in reversed(list(rewards)):
        total = float(reward) + float(gamma) * total
    return float(total)


def _save_train_curve(values: Sequence[float], market: str, cfg: argparse.Namespace) -> str:
    out_dir = _get_results_out_dir(market, cfg)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"regime_moe_train_cumred_{cfg.seed}_{cfg.num_sectors}_{cfg.timestamp}.csv")
    rows = [{"epoch": i, "train_cumred": float(v)} for i, v in enumerate(values, start=1)]
    return write_csv_rows(path, fieldnames=["epoch", "train_cumred"], rows=rows)


def _save_refit_curve(rows: Sequence[Dict[str, Any]], market: str, cfg: argparse.Namespace) -> str:
    out_dir = _get_results_out_dir(market, cfg)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"regime_moe_refit_{cfg.seed}_{cfg.num_sectors}_{cfg.timestamp}.csv")
    return write_csv_rows(
        path,
        fieldnames=["refit_epoch", "discounted_reward", "cumret"],
        rows=rows,
    )


def _save_eval_outputs(
    amount: np.ndarray,
    monthly_sector_metrics: Optional[List[dict]],
    market: str,
    cfg: argparse.Namespace,
    suffix: str = "",
) -> Tuple[str, str]:
    out_dir = _get_results_out_dir(market, cfg)
    os.makedirs(out_dir, exist_ok=True)
    suffix_part = f"_{suffix}" if suffix else ""
    amount_path = os.path.join(out_dir, f"regime_moe_{cfg.seed}_{cfg.num_sectors}_{cfg.timestamp}{suffix_part}.csv")
    monthly_path = os.path.join(out_dir, f"regime_moe_monthly_sector_{cfg.seed}_{cfg.num_sectors}_{cfg.timestamp}{suffix_part}.csv")
    save_amount_curve_csv(amount_path, amount, normalize=True)
    save_monthly_sector_metrics_csv(monthly_sector_metrics, monthly_path)
    return amount_path, monthly_path


def _save_diagnostics(rows: Sequence[Dict[str, Any]], market: str, cfg: argparse.Namespace, suffix: str) -> str:
    out_dir = _get_results_out_dir(market, cfg)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"regime_moe_router_{cfg.seed}_{cfg.num_sectors}_{cfg.timestamp}_{suffix}.csv")
    fieldnames = [
        "step",
        "day",
        "absolute_day",
        "date",
        "prior_source",
        "q_bull",
        "q_bear",
        "q_neutral",
        "confidence",
        "g_bull",
        "g_bear",
        "g_neutral",
        "rho",
        "h_bull",
        "h_bear",
        "h_neutral",
        "reward",
        "paper_bull",
        "paper_bear",
        "paper_neutral",
        "portfolio_cash",
        "portfolio_hhi",
        "bull_cash",
        "bull_risk_exposure",
        "bull_hhi",
        "bull_effective_n",
        "bull_momentum_exposure",
        "bull_cvar_exposure",
        "bull_downside_exposure",
        "bear_cash",
        "bear_risk_exposure",
        "bear_hhi",
        "bear_effective_n",
        "bear_momentum_exposure",
        "bear_cvar_exposure",
        "bear_downside_exposure",
        "neutral_cash",
        "neutral_risk_exposure",
        "neutral_hhi",
        "neutral_effective_n",
        "neutral_momentum_exposure",
        "neutral_cvar_exposure",
        "neutral_downside_exposure",
    ]
    return write_csv_rows(path, fieldnames=fieldnames, rows=rows)


def _save_action_archive(result: TrajectoryResult, market: str, cfg: argparse.Namespace, suffix: str) -> str:
    out_dir = _get_results_out_dir(market, cfg)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"regime_moe_actions_{cfg.seed}_{cfg.num_sectors}_{cfg.timestamp}_{suffix}.npz")
    np.savez_compressed(
        path,
        expert_actions=np.asarray(result.expert_actions, dtype=np.float32),
        portfolio_actions=np.asarray(result.portfolio_actions, dtype=np.float32),
        regime_names=np.asarray(REGIME_NAMES, dtype="U16"),
        absolute_days=np.asarray([row["absolute_day"] for row in result.diagnostics], dtype=np.int64),
        dates=np.asarray([row["date"] for row in result.diagnostics], dtype="U16"),
    )
    return path


def _action_characteristics(action: np.ndarray, spec_signals: Dict[str, np.ndarray]) -> Dict[str, float]:
    weights = np.asarray(action, dtype=np.float64).reshape(-1)
    cash = float(weights[0]) if weights.size else 1.0
    stocks = weights[1:]
    risk_exposure = float(np.sum(stocks))
    stock_hhi = float(np.sum(stocks * stocks))
    effective_n = float(risk_exposure * risk_exposure / (stock_hhi + 1e-12)) if risk_exposure > 1e-12 else 0.0
    return {
        "cash": cash,
        "risk_exposure": risk_exposure,
        "hhi": float(np.sum(weights * weights)),
        "effective_n": effective_n,
        "momentum_exposure": float(np.dot(weights, np.asarray(spec_signals["momentum"], dtype=np.float64))),
        "cvar_exposure": float(np.dot(weights, np.asarray(spec_signals["cvar"], dtype=np.float64))),
        "downside_exposure": float(np.dot(weights, np.asarray(spec_signals["downside"], dtype=np.float64))),
    }


def _phase_for_epoch(epoch: int, cfg: argparse.Namespace) -> str:
    if cfg.ablation_mode == "single_expert":
        return "joint"
    if cfg.ablation_mode == "llm_only" and epoch <= int(cfg.expert_pretrain_epochs) + int(cfg.router_pretrain_epochs):
        return "expert_pretrain"
    if epoch <= int(cfg.expert_pretrain_epochs):
        return "expert_pretrain"
    if epoch <= int(cfg.expert_pretrain_epochs) + int(cfg.router_pretrain_epochs):
        return "router_pretrain"
    return "joint"


def _joint_kl_mult(epoch: int, cfg: argparse.Namespace) -> float:
    joint_start = int(cfg.expert_pretrain_epochs) + int(cfg.router_pretrain_epochs) + 1
    joint_epochs = max(int(cfg.epochs) - joint_start + 1, 1)
    progress = float(max(int(epoch) - joint_start, 0)) / float(max(joint_epochs - 1, 1))
    return float(cfg.joint_kl_final_mult + (1.0 - cfg.joint_kl_final_mult) * (1.0 - progress))


def _transition_from_step(
    *,
    action_info: Any,
    reward: float,
    done: bool,
    paper_rewards: np.ndarray,
    spec_signals: Dict[str, np.ndarray],
    asset_returns: np.ndarray,
    prev_expert_weights: np.ndarray,
    prev_portfolio_weights: np.ndarray,
) -> RegimeMoETransition:
    return RegimeMoETransition(
        expert_state=action_info.expert_state,
        router_input=action_info.router_input,
        market_features=action_info.market_features,
        prior=action_info.prior,
        confidence=float(action_info.confidence),
        reliability=action_info.reliability,
        causal_prior=action_info.causal_prior,
        router_weights=action_info.router_weights,
        expert_actions=action_info.expert_actions,
        expert_policy_actions=action_info.expert_policy_actions,
        expert_anchors=action_info.expert_anchors,
        expert_logp=action_info.expert_logp,
        expert_entropy=action_info.expert_entropy,
        joint_logp=float(action_info.joint_logp),
        value=float(action_info.value),
        reward=float(reward),
        done=1.0 if done else 0.0,
        paper_rewards=np.asarray(paper_rewards, dtype=np.float32),
        spec_momentum=np.asarray(spec_signals["momentum"], dtype=np.float32),
        spec_cvar=np.asarray(spec_signals["cvar"], dtype=np.float32),
        spec_downside=np.asarray(spec_signals["downside"], dtype=np.float32),
        asset_returns=np.asarray(asset_returns, dtype=np.float32),
        prev_expert_weights=np.asarray(prev_expert_weights, dtype=np.float32),
        prev_portfolio_weights=np.asarray(prev_portfolio_weights, dtype=np.float32),
    )


@torch.no_grad()
def collect_trajectory(
    *,
    agent: RegimeMoEPortfolioAgent,
    env: StockTradingEnv,
    prior_provider: RegimePriorProvider,
    cfg: argparse.Namespace,
    deterministic: bool,
    max_steps: int,
    keep_transitions: bool,
    reliability: Optional[EWMAExpertReliability] = None,
) -> TrajectoryResult:
    agent.train_mode(False if deterministic else True)
    if reliability is None:
        reliability = EWMAExpertReliability(
            num_experts=len(REGIME_NAMES),
            action_dim=int(env.shares_num) + 1,
            beta=cfg.reliability_beta,
            transaction_cost=cfg.paper_transaction_cost,
        )
    state = env.reset()
    amount = [float(env.total_asset)]
    rewards: List[float] = []
    raw_returns: List[float] = []
    transitions: List[RegimeMoETransition] = []
    diagnostics: List[Dict[str, Any]] = []
    expert_action_rows: List[np.ndarray] = []
    portfolio_action_rows: List[np.ndarray] = []
    info: Dict[str, Any] = {}
    steps = max(int(max_steps), 1)

    for step in range(steps):
        day_t = int(env.day)
        prior = prior_provider.for_day(env, day_t)
        market_features = compute_policy_market_features(
            env,
            day_t,
            profile=cfg.asset_feature_profile,
        )
        h_t = reliability.scores()
        expert_anchors = None
        if cfg.ablation_mode != "single_expert":
            expert_anchors = compute_causal_expert_anchors(
                env,
                day_t,
                style_strength=cfg.anchor_style_strength,
            )
        asset_features = compute_asset_features(
            env,
            day_t,
            profile=cfg.asset_feature_profile,
        )
        augmented_state = np.concatenate([np.asarray(state, dtype=np.float32), asset_features])
        action, action_info = agent.act(
            env_state=augmented_state,
            market_features=market_features,
            prior=prior.q,
            confidence=prior.confidence,
            reliability=h_t,
            expert_anchors=expert_anchors,
            causal_prior=heuristic_regime_weights(market_features),
            deterministic=deterministic,
        )
        spec_signals = compute_specialization_signals(env, day_t, action_dim=int(env.shares_num) + 1)
        full_prices = np.asarray(getattr(env, "full_close_ary", env.close_ary), dtype=np.float64)
        absolute_t = int(getattr(env, "beg_idx", 0)) + day_t
        absolute_t1 = min(absolute_t + 1, full_prices.shape[0] - 1)
        asset_returns = np.concatenate(
            [[0.0], full_prices[absolute_t1] / np.maximum(full_prices[absolute_t], 1e-12) - 1.0]
        )
        prev_expert_weights = np.asarray(reliability.prev_post_weights, dtype=np.float32).copy()
        prev_portfolio_weights = np.asarray(env._portfolio_weights_full(env.close_ary[day_t]), dtype=np.float32)
        next_state, reward, done, info = env.step(action)
        paper_rewards = reliability.update_from_realized_return(env, day_t, action_info.expert_actions)
        terminal = bool(done or step == steps - 1)
        absolute_day = int(getattr(env, "beg_idx", 0)) + day_t
        slice_dates = tuple(getattr(env, "slice_dates", ()) or ())
        date = str(slice_dates[day_t]) if 0 <= day_t < len(slice_dates) else str(absolute_day)
        expert_characteristics = [
            _action_characteristics(action_info.expert_actions[k], spec_signals)
            for k in range(len(REGIME_NAMES))
        ]
        portfolio_characteristics = _action_characteristics(action, spec_signals)

        if keep_transitions:
            transitions.append(
                _transition_from_step(
                    action_info=action_info,
                    reward=float(reward),
                    done=terminal,
                    paper_rewards=paper_rewards,
                    spec_signals=spec_signals,
                    asset_returns=asset_returns,
                    prev_expert_weights=prev_expert_weights,
                    prev_portfolio_weights=prev_portfolio_weights,
                )
            )
        diagnostics.append(
            {
                "step": int(step),
                "day": int(day_t),
                "absolute_day": absolute_day,
                "date": date,
                "prior_source": prior.source,
                "q_bull": float(prior.q[0]),
                "q_bear": float(prior.q[1]),
                "q_neutral": float(prior.q[2]),
                "confidence": float(prior.confidence),
                "g_bull": float(action_info.router_weights[0]),
                "g_bear": float(action_info.router_weights[1]),
                "g_neutral": float(action_info.router_weights[2]),
                "rho": float(action_info.rho),
                "h_bull": float(h_t[0]),
                "h_bear": float(h_t[1]),
                "h_neutral": float(h_t[2]),
                "reward": float(reward),
                "paper_bull": float(paper_rewards[0]),
                "paper_bear": float(paper_rewards[1]),
                "paper_neutral": float(paper_rewards[2]),
                "portfolio_cash": portfolio_characteristics["cash"],
                "portfolio_hhi": portfolio_characteristics["hhi"],
                **{
                    f"{name.lower()}_{metric}": float(expert_characteristics[k][metric])
                    for k, name in enumerate(REGIME_NAMES)
                    for metric in (
                        "cash",
                        "risk_exposure",
                        "hhi",
                        "effective_n",
                        "momentum_exposure",
                        "cvar_exposure",
                        "downside_exposure",
                    )
                },
            }
        )
        expert_action_rows.append(np.asarray(action_info.expert_actions, dtype=np.float32))
        portfolio_action_rows.append(np.asarray(action, dtype=np.float32))

        amount.append(float(env.total_asset))
        rewards.append(float(reward))
        raw_returns.append(float(info.get("raw_return", reward)))
        state = next_state
        if done:
            break

    amount_arr = np.asarray(amount, dtype=np.float64)
    peaks = np.maximum.accumulate(amount_arr)
    drawdowns = 1.0 - amount_arr / (peaks + 1e-12)
    stats = {
        "raw_ret_sum": float(np.sum(raw_returns)),
        "vol": float(np.std(raw_returns)) if len(raw_returns) >= 2 else 0.0,
        "max_drawdown": float(np.max(drawdowns)) if drawdowns.size > 0 else 0.0,
        "monthly_sector_metrics": info.get("monthly_sector_metrics"),
    }
    cumret_info = info.get("cumret") if isinstance(info, dict) else None
    if cumret_info is None and len(amount_arr) > 0:
        cumret_info = amount_arr[-1] / (amount_arr[0] + 1e-12)
    cumret = float(cumret_info) if cumret_info is not None else float("nan")
    return TrajectoryResult(
        transitions=transitions,
        amount=amount_arr,
        reward_sum=float(np.sum(rewards)),
        discounted_reward=_compute_discounted_return(rewards, cfg.gamma),
        cumret=cumret,
        stats=stats,
        diagnostics=diagnostics,
        expert_actions=np.asarray(expert_action_rows, dtype=np.float32),
        portfolio_actions=np.asarray(portfolio_action_rows, dtype=np.float32),
    )


def evaluate_regime_moe(
    *,
    agent: RegimeMoEPortfolioAgent,
    env: StockTradingEnv,
    prior_provider: RegimePriorProvider,
    cfg: argparse.Namespace,
    reliability: Optional[EWMAExpertReliability] = None,
) -> TrajectoryResult:
    old_random_reset = bool(getattr(env, "if_random_reset", False))
    env.if_random_reset = False
    try:
        return collect_trajectory(
            agent=agent,
            env=env,
            prior_provider=prior_provider,
            cfg=cfg,
            deterministic=True,
            max_steps=max(int(env.max_step) - 1, 1),
            keep_transitions=False,
            reliability=reliability,
        )
    finally:
        env.if_random_reset = old_random_reset


def _build_agent(env: StockTradingEnv, cfg: argparse.Namespace) -> RegimeMoEPortfolioAgent:
    state = env.reset()
    market_features = compute_policy_market_features(
        env,
        int(env.day),
        profile=cfg.asset_feature_profile,
    )
    asset_features = compute_asset_features(
        env,
        int(env.day),
        profile=cfg.asset_feature_profile,
    )
    expert_state_dim = (
        int(np.asarray(state, dtype=np.float32).reshape(-1).size)
        + int(asset_features.size)
        + int(market_features.size)
    )
    router_input_dim = int(market_features.size)
    if cfg.ablation_mode not in {"no_llm_prior", "single_expert"}:
        router_input_dim += len(REGIME_NAMES)
    if cfg.ablation_mode not in {"no_llm_prior", "no_confidence", "single_expert"}:
        router_input_dim += 1
    if cfg.ablation_mode != "single_expert":
        router_input_dim += len(REGIME_NAMES)
    agent_cfg = RegimeMoEConfig(
        expert_state_dim=expert_state_dim,
        router_input_dim=router_input_dim,
        action_dim=int(env.shares_num) + 1,
        market_feature_dim=int(market_features.size),
        hidden_sizes=tuple(int(x) for x in cfg.hidden_sizes),
        router_hidden_sizes=tuple(int(x) for x in cfg.router_hidden_sizes),
        device=cfg.device,
        gamma=cfg.gamma,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        router_lr=cfg.router_lr,
        batch_size=cfg.batch_size,
        ppo_epochs=cfg.ppo_epochs,
        clip_ratio=cfg.clip_ratio,
        ent_coef=cfg.ent_coef,
        value_coef=cfg.value_coef,
        max_grad_norm=cfg.max_grad_norm,
        lambda_spec=cfg.lambda_spec,
        lambda_lb=cfg.lambda_lb,
        router_kl_coef=cfg.router_kl_coef,
        router_reliability_coef=cfg.router_reliability_coef,
        router_reward_coef=cfg.router_reward_coef,
        expert_responsibility_alpha=cfg.expert_responsibility_alpha,
        min_concentration=cfg.min_concentration,
        expert_logit_scale=cfg.expert_logit_scale,
        direct_return_coef=cfg.direct_return_coef,
        direct_downside_coef=cfg.direct_downside_coef,
        paper_transaction_cost=cfg.paper_transaction_cost,
        asset_feature_start=int(np.asarray(state, dtype=np.float32).reshape(-1).size),
        asset_feature_dim=int(asset_features.size // max(int(env.shares_num), 1)),
        asset_scorer_type=cfg.asset_scorer_type,
        asset_scorer_init=cfg.asset_scorer_init,
        cash_scorer_init=cfg.cash_scorer_init,
        portfolio_top_k=cfg.portfolio_top_k,
        anchor_strength=cfg.anchor_strength,
        router_causal_kl_coef=cfg.router_causal_kl_coef,
        causal_prior_strength=cfg.causal_prior_strength,
        llm_prior_strength=cfg.llm_prior_strength,
        no_confidence_fixed_value=cfg.no_confidence_fixed_value,
        ablation_mode=cfg.ablation_mode,
    )
    LOGGER.info(
        "[Policy] regimes=%s | action_dim=%d | expert_state_dim=%d | router_input_dim=%d | market_feature_dim=%d",
        ",".join(REGIME_NAMES),
        agent_cfg.action_dim,
        agent_cfg.expert_state_dim,
        agent_cfg.router_input_dim,
        agent_cfg.market_feature_dim,
    )
    return RegimeMoEPortfolioAgent(agent_cfg)


def train_regime_moe(cfg: argparse.Namespace) -> None:
    cfg.device = _resolve_device(cfg.device)
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    full_env = make_env(0, 10**9, "full", cfg)
    full_env.reset()
    if cfg.write_prior_cache:
        provider_for_cache = RegimePriorProvider(
            market=cfg.market,
            data_dir=os.path.join(PROJECT_DIR, "data"),
            cache_path=None,
            mode="heuristic",
            news_lookback_days=cfg.news_lookback_days,
        )
        cache_path = provider_for_cache.write_heuristic_cache(full_env, cfg.write_prior_cache)
        LOGGER.info("[Prior] wrote heuristic prior cache: %s", cache_path)

    train_beg, train_end, val_beg, val_end, test_beg, test_end = split_622(int(full_env.max_step))
    LOGGER.info(
        "[Split] Total=%d | Train:[%d,%d) Val:[%d,%d) Test:[%d,%d)",
        int(full_env.max_step),
        train_beg,
        train_end,
        val_beg,
        val_end,
        test_beg,
        test_end,
    )

    refit_probe_beg: Optional[int] = None
    if cfg.validation_refit_probe:
        refit_probe_beg = val_beg + max((val_end - val_beg) // 2, 1)
        if not val_beg < refit_probe_beg < val_end:
            raise ValueError("validation window is too short for a chronological refit probe")
    train_env = make_env(train_beg, train_end, "train", cfg)
    val_env = make_env(
        val_beg,
        refit_probe_beg if refit_probe_beg is not None else val_end,
        "val_selection" if refit_probe_beg is not None else "val",
        cfg,
    )
    test_env = None if cfg.validation_only else make_env(test_beg, test_end, "test", cfg)
    cfg.num_sectors = int(getattr(train_env, "sector_num", cfg.num_sectors))
    prior_provider = RegimePriorProvider(
        market=cfg.market,
        data_dir=os.path.join(PROJECT_DIR, "data"),
        cache_path=cfg.prior_cache,
        mode=cfg.prior_mode,
        news_lookback_days=cfg.news_lookback_days,
    )
    cfg.no_confidence_fixed_value = 1.0
    agent = _build_agent(train_env, cfg)
    LOGGER.info(
        "[Prior] mode=%s cache=%s loaded_days=%d news=%s",
        cfg.prior_mode,
        cfg.prior_cache or "<none>",
        len(prior_provider.cache_by_abs_day),
        "yes" if prior_provider.news_index is not None else "no",
    )
    manifest_path = _save_manifest(
        cfg,
        status="running",
        extra={
            "split": {
                "train": [train_beg, train_end],
                "validation": [val_beg, val_end],
                "test": [test_beg, test_end],
            }
        },
    )
    LOGGER.info("[Manifest] running manifest: %s", manifest_path)

    full_len_train = max(int(train_env.max_step) - 1, 1)
    steps_per_epoch = int(cfg.steps_per_epoch) if int(cfg.steps_per_epoch) > 0 else full_len_train
    best_selection_score = -float("inf")
    best_val_cumret = float("nan")
    best_epoch = 0
    best_state: Optional[dict] = None
    best_val_metrics: Dict[str, float] = {}
    validation_history: List[Dict[str, Any]] = []
    train_curve: List[float] = []
    train_curve_path = ""
    start_time = time.time()

    for epoch in range(1, int(cfg.epochs) + 1):
        phase = _phase_for_epoch(epoch, cfg)
        traj = collect_trajectory(
            agent=agent,
            env=train_env,
            prior_provider=prior_provider,
            cfg=cfg,
            deterministic=False,
            max_steps=steps_per_epoch,
            keep_transitions=True,
        )

        if phase == "expert_pretrain":
            metrics = agent.update_expert_pretrain(traj.transitions)
        elif phase == "router_pretrain":
            metrics = agent.update_router_pretrain(traj.transitions)
        else:
            metrics = agent.update_joint(traj.transitions, kl_coef_mult=_joint_kl_mult(epoch, cfg))

        train_curve.append(float(traj.discounted_reward))
        train_curve_path = _save_train_curve(train_curve, train_env.market, cfg)

        val_cumret = float("nan")
        if epoch % max(int(cfg.eval_every), 1) == 0:
            eval_reliability = EWMAExpertReliability(
                num_experts=len(REGIME_NAMES),
                action_dim=int(train_env.shares_num) + 1,
                beta=cfg.reliability_beta,
                transaction_cost=cfg.paper_transaction_cost,
            )
            val_result = evaluate_regime_moe(
                agent=agent,
                env=val_env,
                prior_provider=prior_provider,
                cfg=cfg,
                reliability=eval_reliability,
            )
            val_cumret = float(val_result.cumret)
            val_metrics = _portfolio_metrics(val_result.amount)
            # The staged method is not fully instantiated until joint training.
            # Selecting an untrained/router-only checkpoint can yield a superficially
            # good validation curve while discarding the learned routing and expert
            # specialization required by the method.
            eligible_for_selection = phase == "joint"
            selection_score = _validation_selection_score(
                val_cumret,
                val_metrics,
                cfg.checkpoint_selection_metric,
            )
            validation_history.append(
                {
                    "epoch": int(epoch),
                    "phase": str(phase),
                    "eligible_for_selection": bool(eligible_for_selection),
                    "selection_metric": str(cfg.checkpoint_selection_metric),
                    "selection_score": float(selection_score),
                    "cumret": float(val_cumret),
                    **{key: float(val_metrics[key]) for key in ("ARR", "SR", "SOR", "CR", "MDD")},
                }
            )
            if eligible_for_selection and selection_score > best_selection_score:
                best_selection_score = selection_score
                best_val_cumret = val_cumret
                best_epoch = int(epoch)
                best_state = copy.deepcopy(agent.state_dict())
                best_val_metrics = dict(val_metrics)
                LOGGER.info(
                    "[Eval] New best epoch=%d | selection=%s score=%.6f | Val Cumx=%.4f | ARR=%.6f SR=%.6f SOR=%.6f CR=%.6f",
                    epoch,
                    cfg.checkpoint_selection_metric,
                    selection_score,
                    val_cumret,
                    val_metrics["ARR"],
                    val_metrics["SR"],
                    val_metrics["SOR"],
                    val_metrics["CR"],
                )

        elapsed = (time.time() - start_time) / 60.0
        metric_text = " ".join(f"{key}={float(value):.4f}" for key, value in sorted(metrics.items()))
        LOGGER.info(
            "Epoch %04d | phase=%s | steps=%d | TrainObj=%.4f | TrainCumx=%.4f | ValCumx=%.4f | %s | %.1f min",
            epoch,
            phase,
            len(traj.transitions),
            traj.discounted_reward,
            traj.cumret,
            val_cumret,
            metric_text,
            elapsed,
        )

    if best_state is not None:
        agent.load_state_dict(best_state)
    else:
        best_epoch = int(cfg.epochs)
        best_selection_score = float("nan")
        best_val_cumret = float("nan")

    final_reliability = EWMAExpertReliability(
        num_experts=len(REGIME_NAMES),
        action_dim=int(train_env.shares_num) + 1,
        beta=cfg.reliability_beta,
        transaction_cost=cfg.paper_transaction_cost,
    )
    final_val = evaluate_regime_moe(
        agent=agent,
        env=val_env,
        prior_provider=prior_provider,
        cfg=cfg,
        reliability=final_reliability,
    )
    final_val_metrics = _portfolio_metrics(final_val.amount)
    if cfg.validation_refit_probe:
        if refit_probe_beg is None:
            raise RuntimeError("internal refit-probe boundary is unavailable")
        refit_env = make_env(train_beg, refit_probe_beg, "validation_refit_train", cfg)
        probe_env = make_env(refit_probe_beg, val_end, "validation_refit_probe", cfg)
        refit_steps = max(int(refit_env.max_step) - 1, 1)
        refit_history: List[Dict[str, Any]] = []
        probe_history: List[Dict[str, Any]] = []
        best_probe_score = -float("inf")
        best_probe_epoch = 0
        # Evaluate the validation-selected checkpoint before any optional
        # train+selection-window refit.  Omitting epoch 0 silently forced every
        # candidate through at least ``eval_every`` additional updates, even
        # when those updates degraded the later chronological probe.  Epoch 0
        # is a legitimate, fully causal refit choice and is needed to assess
        # whether refitting helps at all.
        initial_probe = evaluate_regime_moe(
            agent=agent,
            env=probe_env,
            prior_provider=prior_provider,
            cfg=cfg,
        )
        initial_probe_metrics = _portfolio_metrics(initial_probe.amount)
        initial_probe_score = _validation_selection_score(
            initial_probe.cumret,
            initial_probe_metrics,
            cfg.checkpoint_selection_metric,
        )
        probe_history.append(
            {
                "refit_epoch": 0,
                "selection_metric": str(cfg.checkpoint_selection_metric),
                "selection_score": float(initial_probe_score),
                "cumret": float(initial_probe.cumret),
                **{key: float(initial_probe_metrics[key]) for key in ("ARR", "SR", "SOR", "CR", "MDD")},
            }
        )
        best_probe_score = float(initial_probe_score)
        best_probe_state: Optional[dict] = copy.deepcopy(agent.state_dict())
        probe_start = time.time()
        for refit_epoch in range(1, int(cfg.refit_epochs) + 1):
            refit_traj = collect_trajectory(
                agent=agent,
                env=refit_env,
                prior_provider=prior_provider,
                cfg=cfg,
                deterministic=False,
                max_steps=refit_steps,
                keep_transitions=True,
            )
            refit_update = agent.update_joint(
                refit_traj.transitions,
                kl_coef_mult=float(cfg.joint_kl_final_mult),
            )
            refit_history.append(
                {
                    "refit_epoch": int(refit_epoch),
                    "discounted_reward": float(refit_traj.discounted_reward),
                    "cumret": float(refit_traj.cumret),
                }
            )
            if refit_epoch % max(int(cfg.eval_every), 1) == 0:
                probe_result = evaluate_regime_moe(
                    agent=agent,
                    env=probe_env,
                    prior_provider=prior_provider,
                    cfg=cfg,
                )
                probe_metrics = _portfolio_metrics(probe_result.amount)
                probe_score = _validation_selection_score(
                    probe_result.cumret,
                    probe_metrics,
                    cfg.checkpoint_selection_metric,
                )
                probe_history.append(
                    {
                        "refit_epoch": int(refit_epoch),
                        "selection_metric": str(cfg.checkpoint_selection_metric),
                        "selection_score": float(probe_score),
                        "cumret": float(probe_result.cumret),
                        **{key: float(probe_metrics[key]) for key in ("ARR", "SR", "SOR", "CR", "MDD")},
                    }
                )
                if probe_score > best_probe_score:
                    best_probe_score = float(probe_score)
                    best_probe_epoch = int(refit_epoch)
                    best_probe_state = copy.deepcopy(agent.state_dict())
            LOGGER.info(
                "Validation refit probe %04d/%04d | steps=%d | TrainObj=%.4f | TrainCumx=%.4f | %s | %.1f min",
                refit_epoch,
                int(cfg.refit_epochs),
                len(refit_traj.transitions),
                refit_traj.discounted_reward,
                refit_traj.cumret,
                " ".join(f"{key}={float(value):.4f}" for key, value in sorted(refit_update.items())),
                (time.time() - probe_start) / 60.0,
            )
        if best_probe_state is None:
            raise RuntimeError("validation refit probe produced no eligible evaluation")
        agent.load_state_dict(best_probe_state)
        final_probe = evaluate_regime_moe(
            agent=agent,
            env=probe_env,
            prior_provider=prior_provider,
            cfg=cfg,
        )
        final_probe_metrics = _portfolio_metrics(final_probe.amount)
        probe_amount_path, probe_monthly_path = _save_eval_outputs(
            final_probe.amount,
            final_probe.stats.get("monthly_sector_metrics"),
            probe_env.market,
            cfg,
            suffix="final_refit_probe",
        )
        probe_diag_path = _save_diagnostics(
            final_probe.diagnostics, probe_env.market, cfg, suffix="final_refit_probe"
        )
        probe_action_path = _save_action_archive(
            final_probe, probe_env.market, cfg, suffix="final_refit_probe"
        )
        refit_curve_path = _save_refit_curve(refit_history, train_env.market, cfg)
        if cfg.ckpt:
            ckpt_path = (
                os.path.join(_get_results_out_dir(cfg.market, cfg), f"model_{cfg.seed}_{cfg.timestamp}.pt")
                if str(cfg.ckpt).strip().lower() == "auto"
                else (os.path.join(PROJECT_DIR, cfg.ckpt) if not os.path.isabs(cfg.ckpt) else cfg.ckpt)
            )
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(agent.state_dict(), ckpt_path)
        else:
            ckpt_path = ""
        manifest_path = _save_manifest(
            cfg,
            status="complete",
            extra={
                "evaluation_scope": "validation_refit_probe",
                "test_evaluated": False,
                "split": {
                    "train": [train_beg, train_end],
                    "validation_selection": [val_beg, refit_probe_beg],
                    "validation_refit_train": [train_beg, refit_probe_beg],
                    "validation_refit_probe": [refit_probe_beg, val_end],
                    "test": [test_beg, test_end],
                },
                "selection": {
                    "best_epoch": int(best_epoch),
                    "checkpoint_selection_metric": str(cfg.checkpoint_selection_metric),
                    "best_validation_score": float(best_selection_score),
                    "best_validation_cumret": float(best_val_cumret),
                    "best_validation_metrics": best_val_metrics,
                    "validation_history": validation_history,
                },
                "validation_selection_metrics": final_val_metrics,
                "refit_probe": {
                    "best_refit_epoch": int(best_probe_epoch),
                    "best_probe_score": float(best_probe_score),
                    "probe_history": probe_history,
                    "final_probe_metrics": final_probe_metrics,
                    "test_accessed": False,
                },
                "refit": {
                    "enabled": True,
                    "epochs": int(cfg.refit_epochs),
                    "range": [train_beg, refit_probe_beg],
                    "test_accessed_during_refit": False,
                    "update_phase": "joint",
                    "history": refit_history,
                },
                "outputs": {
                    "validation_probe_amount": probe_amount_path,
                    "validation_probe_monthly": probe_monthly_path,
                    "validation_probe_router_diagnostics": probe_diag_path,
                    "validation_probe_expert_actions": probe_action_path,
                    "checkpoint": ckpt_path,
                    "train_curve": train_curve_path,
                    "refit_curve": refit_curve_path,
                    "log": str(getattr(cfg, "log_path", "")),
                },
            },
        )
        LOGGER.info("[Manifest] validation refit probe complete: %s", manifest_path)
        LOGGER.info(
            "[Validation refit probe] Best refit epoch=%d | ARR=%.6f SR=%.6f SOR=%.6f CR=%.6f; test not evaluated",
            best_probe_epoch,
            final_probe_metrics["ARR"],
            final_probe_metrics["SR"],
            final_probe_metrics["SOR"],
            final_probe_metrics["CR"],
        )
        return

    if cfg.validation_only:
        validation_amount_path, validation_monthly_path = _save_eval_outputs(
            final_val.amount,
            final_val.stats.get("monthly_sector_metrics"),
            val_env.market,
            cfg,
            suffix="final_validation",
        )
        validation_diag_path = _save_diagnostics(
            final_val.diagnostics, val_env.market, cfg, suffix="final_validation"
        )
        validation_action_path = _save_action_archive(
            final_val, val_env.market, cfg, suffix="final_validation"
        )
        if cfg.ckpt:
            if str(cfg.ckpt).strip().lower() == "auto":
                ckpt_path = os.path.join(
                    _get_results_out_dir(cfg.market, cfg),
                    f"model_{cfg.seed}_{cfg.timestamp}.pt",
                )
            else:
                ckpt_path = os.path.join(PROJECT_DIR, cfg.ckpt) if not os.path.isabs(cfg.ckpt) else cfg.ckpt
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(agent.state_dict(), ckpt_path)
        else:
            ckpt_path = ""
        manifest_path = _save_manifest(
            cfg,
            status="complete",
            extra={
                "evaluation_scope": "validation_only",
                "test_evaluated": False,
                "split": {
                    "train": [train_beg, train_end],
                    "validation": [val_beg, val_end],
                    "test": [test_beg, test_end],
                },
                "selection": {
                    "best_epoch": int(best_epoch),
                    "checkpoint_selection_metric": str(cfg.checkpoint_selection_metric),
                    "best_validation_score": float(best_selection_score),
                    "best_validation_cumret": float(best_val_cumret),
                    "best_validation_metrics": best_val_metrics,
                    "validation_history": validation_history,
                },
                "validation_metrics": final_val_metrics,
                "outputs": {
                    "validation_amount": validation_amount_path,
                    "validation_router_diagnostics": validation_diag_path,
                    "validation_expert_actions": validation_action_path,
                    "checkpoint": ckpt_path,
                    "train_curve": train_curve_path,
                    "log": str(getattr(cfg, "log_path", "")),
                },
            },
        )
        LOGGER.info("[Manifest] validation-only complete manifest: %s", manifest_path)
        LOGGER.info(
            "[Validation-only] Best epoch=%d | ARR=%.6f SR=%.6f SOR=%.6f CR=%.6f; test not evaluated",
            best_epoch,
            final_val_metrics["ARR"],
            final_val_metrics["SR"],
            final_val_metrics["SOR"],
            final_val_metrics["CR"],
        )
        return

    if test_env is None:
        raise RuntimeError("test environment is unavailable for final evaluation")

    # Hyperparameters and the checkpoint have already been selected using the
    # validation window above.  A formal run may now continue joint training on
    # the combined train+validation history for a fixed, pre-declared number of
    # epochs.  The held-out test environment is neither traversed nor queried
    # during this refit.
    refit_history: List[Dict[str, Any]] = []
    refit_curve_path = ""
    refit_reliability_warmup: Optional[Dict[str, float]] = None
    if int(cfg.refit_epochs) > 0:
        refit_env = make_env(train_beg, val_end, "refit", cfg)
        refit_steps = max(int(refit_env.max_step) - 1, 1)
        refit_start = time.time()
        for refit_epoch in range(1, int(cfg.refit_epochs) + 1):
            refit_traj = collect_trajectory(
                agent=agent,
                env=refit_env,
                prior_provider=prior_provider,
                cfg=cfg,
                deterministic=False,
                max_steps=refit_steps,
                keep_transitions=True,
            )
            refit_update = agent.update_joint(
                refit_traj.transitions,
                kl_coef_mult=float(cfg.joint_kl_final_mult),
            )
            refit_history.append(
                {
                    "refit_epoch": int(refit_epoch),
                    "discounted_reward": float(refit_traj.discounted_reward),
                    "cumret": float(refit_traj.cumret),
                }
            )
            LOGGER.info(
                "Refit %04d/%04d | steps=%d | TrainObj=%.4f | TrainCumx=%.4f | %s | %.1f min",
                refit_epoch,
                int(cfg.refit_epochs),
                len(refit_traj.transitions),
                refit_traj.discounted_reward,
                refit_traj.cumret,
                " ".join(f"{key}={float(value):.4f}" for key, value in sorted(refit_update.items())),
                (time.time() - refit_start) / 60.0,
            )
        refit_curve_path = _save_refit_curve(refit_history, train_env.market, cfg)

        # Reconstruct only the causal reliability state carried into the test
        # period.  This pass performs no gradient updates; it uses validation
        # observations chronologically and stops before test_beg.
        final_reliability = EWMAExpertReliability(
            num_experts=len(REGIME_NAMES),
            action_dim=int(train_env.shares_num) + 1,
            beta=cfg.reliability_beta,
            transaction_cost=cfg.paper_transaction_cost,
        )
        refit_warmup = evaluate_regime_moe(
            agent=agent,
            env=val_env,
            prior_provider=prior_provider,
            cfg=cfg,
            reliability=final_reliability,
        )
        refit_reliability_warmup = _portfolio_metrics(refit_warmup.amount)

    final_test = evaluate_regime_moe(
        agent=agent,
        env=test_env,
        prior_provider=prior_provider,
        cfg=cfg,
        reliability=final_reliability,
    )
    final_amount_path, final_monthly_path = _save_eval_outputs(
        final_test.amount,
        final_test.stats.get("monthly_sector_metrics"),
        test_env.market,
        cfg,
        suffix="final",
    )
    final_diag_path = _save_diagnostics(final_test.diagnostics, test_env.market, cfg, suffix="final")
    final_action_path = _save_action_archive(final_test, test_env.market, cfg, suffix="final")

    if cfg.ckpt:
        if str(cfg.ckpt).strip().lower() == "auto":
            ckpt_path = os.path.join(_get_results_out_dir(cfg.market, cfg), f"model_{cfg.seed}_{cfg.timestamp}.pt")
        else:
            ckpt_path = os.path.join(PROJECT_DIR, cfg.ckpt) if not os.path.isabs(cfg.ckpt) else cfg.ckpt
        parent = os.path.dirname(ckpt_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(agent.state_dict(), ckpt_path)
        LOGGER.info("[Checkpoint] saved %s", ckpt_path)
    else:
        ckpt_path = ""

    final_metrics = _portfolio_metrics(final_test.amount)
    manifest_path = _save_manifest(
        cfg,
        status="complete",
        extra={
            "split": {
                "train": [train_beg, train_end],
                "validation": [val_beg, val_end],
                "test": [test_beg, test_end],
            },
            "selection": {
                "best_epoch": int(best_epoch),
                "checkpoint_selection_metric": str(cfg.checkpoint_selection_metric),
                "best_validation_score": float(best_selection_score),
                "best_validation_cumret": float(best_val_cumret),
                "best_validation_metrics": best_val_metrics,
                "validation_history": validation_history,
            },
            "validation_metrics": final_val_metrics,
            "test_metrics": final_metrics,
            "refit": {
                "enabled": bool(int(cfg.refit_epochs) > 0),
                "epochs": int(cfg.refit_epochs),
                "range": [train_beg, val_end] if int(cfg.refit_epochs) > 0 else None,
                "test_accessed_during_refit": False,
                "update_phase": "joint",
                "history": refit_history,
                "reliability_warmup_range": [val_beg, val_end]
                if int(cfg.refit_epochs) > 0
                else None,
                "reliability_warmup_metrics": refit_reliability_warmup,
            },
            "outputs": {
                "amount": final_amount_path,
                "router_diagnostics": final_diag_path,
                "expert_actions": final_action_path,
                "checkpoint": ckpt_path,
                "train_curve": train_curve_path,
                "refit_curve": refit_curve_path,
                "log": str(getattr(cfg, "log_path", "")),
            },
        },
    )
    LOGGER.info("[Manifest] complete manifest: %s", manifest_path)
    LOGGER.info(
        "[Metrics] ARR=%.6f SR=%.6f SOR=%.6f CR=%.6f MDD=%.6f",
        final_metrics["ARR"],
        final_metrics["SR"],
        final_metrics["SOR"],
        final_metrics["CR"],
        final_metrics["MDD"],
    )

    LOGGER.info(
        "[Final Eval] Best epoch=%d | Best Val Cumx=%.4f | Final Val Cumx=%.4f | "
        "Test Score=%.4f | Test Cumx=%.4f | Test Vol=%.4f | Test MDD=%.4f | "
        "curve=%s | monthly=%s | router=%s | actions=%s",
        best_epoch,
        best_val_cumret,
        final_val.cumret,
        final_test.reward_sum,
        final_test.cumret,
        final_test.stats["vol"],
        final_test.stats["max_drawdown"],
        final_amount_path,
        final_monthly_path,
        final_diag_path,
        final_action_path,
    )
    LOGGER.info("[Output] train cumulative reward curve: %s", train_curve_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Confidence-calibrated LLM-prior Regime-MoE portfolio training.")
    p.add_argument("--market", type=str, default="LSE")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--num-sectors", type=int, default=3)
    p.add_argument("--context-model", type=str, default=None)
    p.add_argument(
        "--ablation-mode",
        type=str,
        default="full",
        choices=["full", "no_llm_prior", "llm_only", "no_confidence", "single_expert"],
        help="Model variant: full method or one of the four paper ablations.",
    )

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--expert-pretrain-epochs", type=int, default=40)
    p.add_argument("--router-pretrain-epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument(
        "--refit-epochs",
        type=int,
        default=0,
        help=(
            "After validation-only selection, continue joint training on the combined "
            "train+validation history for this many fixed epochs before one test evaluation."
        ),
    )
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument(
        "--checkpoint-selection-metric",
        choices=["cumret", "ARR", "SR", "SOR", "CR"],
        default="cumret",
        help="Validation-only metric used for checkpoint selection.",
    )
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--risk-window", type=int, default=20)
    p.add_argument("--vol-penalty-coef", type=float, default=0.0)
    p.add_argument("--drawdown-penalty-coef", type=float, default=0.0)

    p.add_argument("--actor-lr", type=float, default=2e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--router-lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=None)
    p.add_argument("--router-hidden-sizes", type=int, nargs="+", default=None)
    p.add_argument("--min-concentration", type=float, default=0.05)
    p.add_argument("--expert-logit-scale", type=float, default=1.0)
    p.add_argument(
        "--asset-feature-profile",
        choices=[
            "broad",
            "stable_validation",
            "stable_validation_primary",
            "walkforward_stable",
            "method_specialization",
            "hybrid_specialization",
        ],
        default="broad",
        help="Causal per-asset feature basis used by the shared permutation-equivariant scorer.",
    )
    p.add_argument(
        "--asset-scorer-type",
        choices=["mlp", "linear"],
        default="mlp",
        help="Capacity of the shared permutation-equivariant risky-asset scorer.",
    )
    p.add_argument(
        "--asset-scorer-init",
        choices=["default", "primary_positive", "regime_template"],
        default="default",
        help="Recorded initialization for the shared risky-asset scorer.",
    )
    p.add_argument(
        "--cash-scorer-init",
        choices=[
            "default",
            "regime_aligned",
            "risk_first",
            "template_low",
            "hybrid_aligned",
            "regime_template",
        ],
        default="default",
        help="Recorded cold start for expert cash logits; all parameters remain trainable.",
    )
    p.add_argument(
        "--portfolio-top-k",
        type=int,
        default=0,
        help=(
            "If positive, normalize the Dirichlet mean over only the top-k "
            "learned action logits; zero keeps the dense softmax head."
        ),
    )

    p.add_argument("--lambda-spec", type=float, default=0.05)
    p.add_argument("--lambda-lb", type=float, default=0.05)
    p.add_argument("--router-kl-coef", type=float, default=0.0)
    p.add_argument("--joint-kl-final-mult", type=float, default=0.1)
    p.add_argument("--router-reliability-coef", type=float, default=0.0)
    p.add_argument("--router-reward-coef", type=float, default=0.50)
    p.add_argument("--expert-responsibility-alpha", type=float, default=0.10)
    p.add_argument("--reliability-beta", type=float, default=0.94)
    p.add_argument("--paper-transaction-cost", type=float, default=1e-4)
    p.add_argument("--direct-return-coef", type=float, default=1.0)
    p.add_argument("--direct-downside-coef", type=float, default=0.0)
    p.add_argument("--anchor-strength", type=float, default=0.0)
    p.add_argument("--anchor-style-strength", type=float, default=0.0)
    p.add_argument("--router-causal-kl-coef", type=float, default=0.0)
    p.add_argument("--causal-prior-strength", type=float, default=0.0)
    p.add_argument("--llm-prior-strength", type=float, default=1.0)

    p.add_argument("--prior-mode", type=str, default="auto", choices=["auto", "cache", "heuristic"])
    p.add_argument("--prior-cache", type=str, default="")
    p.add_argument("--write-prior-cache", type=str, default="")
    p.add_argument("--news-lookback-days", type=int, default=1)
    p.add_argument(
        "--validation-only",
        action="store_true",
        help="Train and save validation artifacts without evaluating the held-out test split.",
    )
    p.add_argument(
        "--validation-refit-probe",
        action="store_true",
        help=(
            "Chronologically split the validation window in half, optionally refit on its first half, "
            "and select a fixed refit duration (including epoch 0) on its second half without "
            "constructing a test env."
        ),
    )

    p.add_argument("--deterministic", action="store_true")
    p.set_defaults(deterministic=True)
    p.add_argument("--random-reset", dest="random_reset", action="store_true")
    p.add_argument("--no-random-reset", dest="random_reset", action="store_false")
    p.set_defaults(random_reset=True)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--results_dir_name", type=str, default="results_regime_moe")
    p.add_argument("--run-tag", type=str, default="")
    cfg = p.parse_args()

    if cfg.hidden_sizes is None:
        cfg.hidden_sizes = [128, 64]
    if cfg.router_hidden_sizes is None:
        cfg.router_hidden_sizes = list(cfg.hidden_sizes)
    cfg.expert_pretrain_epochs = max(int(cfg.expert_pretrain_epochs), 0)
    cfg.router_pretrain_epochs = max(int(cfg.router_pretrain_epochs), 0)
    if int(cfg.refit_epochs) < 0:
        p.error("--refit-epochs must be non-negative")
    if cfg.validation_refit_probe:
        cfg.validation_only = True
    if float(cfg.expert_logit_scale) <= 0.0:
        p.error("--expert-logit-scale must be positive")
    if not np.isclose(float(cfg.router_kl_coef), 0.0, atol=1e-12):
        p.error("--router-kl-coef must be 0 to match Method.tex")
    method_zero_fields = {
        "--router-causal-kl-coef": cfg.router_causal_kl_coef,
        "--causal-prior-strength": cfg.causal_prior_strength,
        "--router-reliability-coef": cfg.router_reliability_coef,
        "--direct-downside-coef": cfg.direct_downside_coef,
        "--anchor-strength": cfg.anchor_strength,
        "--anchor-style-strength": cfg.anchor_style_strength,
    }
    invalid_method_fields = {
        name: float(value)
        for name, value in method_zero_fields.items()
        if not np.isclose(float(value), 0.0, atol=1e-12)
    }
    if invalid_method_fields:
        p.error(
            "Method.tex does not define the requested auxiliary priors, risk losses, "
            f"or portfolio anchors: {invalid_method_fields}"
        )
    if not np.isclose(float(cfg.llm_prior_strength), 1.0, atol=1e-12):
        p.error("--llm-prior-strength must be 1 to match Method.tex")
    if cfg.expert_pretrain_epochs + cfg.router_pretrain_epochs >= int(cfg.epochs):
        cfg.expert_pretrain_epochs = max(0, int(0.2 * int(cfg.epochs)))
        cfg.router_pretrain_epochs = max(0, int(0.1 * int(cfg.epochs)))
    cfg.timestamp = ""
    cfg.results_dir_name = (
        str(cfg.results_dir_name).rstrip("/")
        + f"_epo{cfg.epochs}_exp{cfg.expert_pretrain_epochs}_rt{cfg.router_pretrain_epochs}"
        + f"_bs{cfg.batch_size}_lb{cfg.lambda_lb}_sp{cfg.lambda_spec}_kl{cfg.router_kl_coef}"
    )
    if str(cfg.run_tag).strip():
        safe_tag = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(cfg.run_tag).strip())
        cfg.results_dir_name += f"_{safe_tag}"
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    LOGGER, log_path, timestamp = _setup_logger(cfg.market)
    cfg.timestamp = timestamp
    cfg.log_path = log_path
    LOGGER.info("[Logging] Log file: %s", log_path)
    _log_run_config(LOGGER, cfg)
    train_regime_moe(cfg)
