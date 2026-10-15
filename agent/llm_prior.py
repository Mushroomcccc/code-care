from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import requests

from .moe_utils import TitleNewsIndex, news_sentiment_score
from .regime_moe_agent import (
    REGIME_NAMES,
    compute_market_features,
    heuristic_regime_weights,
    sanitize_simplex,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if np.isfinite(out) else float(default)


def _round_sig(value: Any, digits: int = 4) -> float:
    number = _safe_float(value, 0.0)
    if abs(number) <= 1e-12:
        return 0.0
    return float(f"{number:.{int(digits)}g}")


def _stable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable_payload(value[key]) for key in sorted(value, key=lambda item: str(item))}
    if isinstance(value, np.ndarray):
        return _stable_payload(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_stable_payload(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalized_entropy_confidence(probs: Sequence[float]) -> float:
    q = sanitize_simplex(probs, len(REGIME_NAMES), fallback_index=2)
    entropy = -float(np.sum(q * np.log(np.clip(q, 1e-12, 1.0))))
    return float(np.clip(1.0 - entropy / math.log(len(REGIME_NAMES)), 0.0, 1.0))


def parse_regime_distribution(response: Any) -> Optional[np.ndarray]:
    if isinstance(response, dict):
        values = []
        for name in REGIME_NAMES:
            matched = None
            for key, value in response.items():
                if str(key).strip().lower() == name.lower():
                    matched = value
                    break
            values.append(_safe_float(matched, np.nan))
        arr = np.asarray(values, dtype=np.float64)
        if np.all(np.isfinite(arr)):
            return sanitize_simplex(arr, len(REGIME_NAMES), fallback_index=2)
        return None

    if isinstance(response, (list, tuple, np.ndarray)):
        arr = np.asarray(response, dtype=np.float64).reshape(-1)
        if arr.size == len(REGIME_NAMES) and np.all(np.isfinite(arr)):
            return sanitize_simplex(arr, len(REGIME_NAMES), fallback_index=2)
        return None

    if not isinstance(response, str):
        return None
    text = response.strip()
    if not text:
        return None

    candidates = [text]
    dict_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if dict_match:
        candidates.append(dict_match.group(0))
    list_match = re.search(r"\[[^\]]+\]", text, flags=re.DOTALL)
    if list_match:
        candidates.append(list_match.group(0))

    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
            except Exception:
                continue
            q = parse_regime_distribution(parsed)
            if q is not None:
                return q

    keyed_values: Dict[str, float] = {}
    for name in REGIME_NAMES:
        pattern = rf"{name}\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            keyed_values[name] = _safe_float(match.group(1), np.nan)
    if len(keyed_values) == len(REGIME_NAMES):
        return parse_regime_distribution(keyed_values)
    return None


@dataclass(frozen=True)
class LLMRegimePriorConfig:
    model: str = "llama3.1"
    endpoint: Optional[str] = "http://127.0.0.1:6019/api/generate"
    mode: str = "auto"
    request_timeout: float = 40.0
    max_retries: int = 3
    samples: int = 5
    summary_temperature: float = 0.0
    regime_temperature: float = 0.7
    news_lookback_days: int = 1
    max_sector_titles: int = 8
    max_stock_titles: int = 3
    max_stock_items: int = 30
    cache_path: Optional[str] = None
    cache_enabled: bool = True


class LocalRegimeLLMClient:
    """Local LLM client using the same request shape as teacher.py."""

    def __init__(
        self,
        *,
        model: str,
        endpoint: Optional[str],
        request_timeout: float,
        max_retries: int,
    ) -> None:
        self.model = str(model)
        self.endpoint = str(endpoint).strip() if endpoint else None
        self.request_timeout = float(request_timeout)
        self.max_retries = max(int(max_retries), 1)

    def generate(self, prompt: str, *, temperature: float = 0.0) -> Optional[str]:
        if not self.endpoint:
            return None
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": float(temperature)},
        }
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.endpoint, json=payload, timeout=self.request_timeout)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and "response" in data:
                    return str(data["response"]).strip()
                return str(data).strip()
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(min(2.0, 0.25 * (attempt + 1)))
        if last_error is not None:
            return None
        return None


class PersistentLLMPriorCache:
    def __init__(self, cache_path: Optional[str], *, enabled: bool = True) -> None:
        self.enabled = bool(enabled) and bool(cache_path)
        self.path = Path(str(cache_path)).expanduser().resolve() if cache_path else None
        self.payload: Dict[str, Any] = {"version": 1, "items": {}}
        if self.enabled and self.path is not None and self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and isinstance(loaded.get("items"), dict):
                    self.payload = loaded
                    self.payload.setdefault("version", 1)
                    self.payload.setdefault("items", {})
            except Exception:
                self.payload = {"version": 1, "items": {}}

    def make_key(self, *, level: str, identity: Dict[str, Any]) -> str:
        payload = {"level": str(level), "identity": _stable_payload(identity)}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return f"{level}:{digest}"

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        items = self.payload.get("items", {})
        entry = items.get(str(key)) if isinstance(items, dict) else None
        if isinstance(entry, dict):
            return entry.get("value")
        return None

    def put(self, key: str, value: Any, metadata: Dict[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return
        items = self.payload.setdefault("items", {})
        if not isinstance(items, dict):
            items = {}
            self.payload["items"] = items
        items[str(key)] = {
            "value": _stable_payload(value),
            "metadata": _stable_payload(metadata),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(self.payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            tmp_path.replace(self.path)
        except Exception:
            return


class LLMRegimePriorBuilder:
    NEWS_SUMMARY_PROMPT = """
Role: Professional financial analyst.
Task: Summarize strictly causal market news evidence for later regime inference.

Trading day: {date_label}
Sector and stock news:
{news_payload}

Output requirements:
Return a concise evidence-grounded summary in plain text.
Mention market-wide events, sector shocks, stock clusters, and risk catalysts only if supported by the provided titles.
Do not invent facts. Do not use future information.
"""

    INDICATOR_SUMMARY_PROMPT = """
Role: Professional quantitative market analyst.
Task: Summarize strictly causal numerical evidence for later regime inference.

Trading day: {date_label}
Market-level features:
{market_features}

Output requirements:
Return a concise numerical summary in plain text.
Discuss only market return, volatility, and drawdown from the supplied values.
"""

    REGIME_PROMPT = """
Role: Hindsight-blind market regime classifier.
Task: Produce a probability distribution over Bull, Bear, and Neutral using only the causal evidence below.

News evidence summary:
{event_summary}

Numerical evidence summary:
{indicator_summary}

Calibration rules:
- Neutral is the default when evidence is mixed, sparse, local, low-conviction, or conflicting.
- Assign high Bull only when numerical trend/risk evidence is jointly constructive, or when news evidence is broad and market-wide.
- Assign high Bear only when numerical trend/risk evidence is clearly stressed, or when news evidence is broad and market-wide negative.
- If news evidence and numerical evidence point in opposite directions, increase Neutral and avoid an overconfident Bull/Bear distribution unless one side is very strong.

Output requirements:
Return ONLY one raw JSON object with exactly these keys:
{{"Bull": float, "Bear": float, "Neutral": float}}
The values must be nonnegative and sum to 1.
Do not output explanation, markdown, labels, or extra text.
"""

    def __init__(self, cfg: LLMRegimePriorConfig) -> None:
        self.cfg = cfg
        self.mode = str(cfg.mode).strip().lower() or "auto"
        if self.mode not in {"auto", "llm", "heuristic"}:
            self.mode = "auto"
        self.client = LocalRegimeLLMClient(
            model=cfg.model,
            endpoint=cfg.endpoint,
            request_timeout=cfg.request_timeout,
            max_retries=cfg.max_retries,
        )
        self.cache = PersistentLLMPriorCache(cfg.cache_path, enabled=cfg.cache_enabled)

    @staticmethod
    def date_label(env: Any, day: int, news_index: Optional[TitleNewsIndex] = None) -> str:
        absolute_day = int(getattr(env, "beg_idx", 0)) + int(day)
        slice_dates = tuple(getattr(env, "slice_dates", ()) or ())
        if 0 <= int(day) < len(slice_dates):
            return str(slice_dates[int(day)])
        if news_index is not None and 0 <= absolute_day < len(news_index.dates):
            return str(news_index.dates[absolute_day])
        return f"absolute_day={absolute_day}"

    def news_payload(self, env: Any, day: int, news_index: Optional[TitleNewsIndex]) -> Dict[str, Any]:
        absolute_day = int(getattr(env, "beg_idx", 0)) + int(day)
        sector_names = [str(x) for x in list(getattr(env, "sector_names", []))]
        if news_index is None:
            return {"sectors": {name: [] for name in sector_names}, "stocks": []}

        sectors = {
            name: news_index.sector_titles(
                absolute_day,
                name,
                lookback_days=self.cfg.news_lookback_days,
                max_titles=self.cfg.max_sector_titles,
            )
            for name in sector_names
        }
        stock_items: List[Dict[str, Any]] = []
        max_items = max(int(self.cfg.max_stock_items), 0)
        for stock_index in range(int(getattr(env, "shares_num", 0))):
            ticker = news_index.ticker_for_index(stock_index)
            titles = news_index.stock_titles(
                absolute_day,
                ticker,
                lookback_days=self.cfg.news_lookback_days,
                max_titles=self.cfg.max_stock_titles,
            )
            if not titles:
                continue
            stock_items.append({"ticker": ticker, "titles": titles})
            if len(stock_items) >= max_items:
                break
        return {"sectors": sectors, "stocks": stock_items}

    def market_feature_payload(self, env: Any, day: int) -> Dict[str, float]:
        values = compute_market_features(env, day)
        selected = [
            ("market_return_5d", 1),
            ("market_return_21d", 2),
            ("market_return_60d", 3),
            ("market_volatility_21d", 4),
            ("market_volatility_60d", 5),
            ("market_downside_volatility_21d", 6),
            ("market_drawdown_60d", 7),
            ("market_return_horizon_conflict_21d_60d", 8),
            ("market_breadth_positive_21d", 9),
            ("market_return_dispersion_21d", 10),
            ("market_downside_breadth_21d", 11),
        ]
        return {
            name: _round_sig(values[index], digits=4)
            for name, index in selected
            if index < len(values)
        }

    def _cached_text(self, *, level: str, prompt: str, metadata: Dict[str, Any], temperature: float) -> Optional[str]:
        key = self.cache.make_key(
            level=level,
            identity={
                "model": self.cfg.model,
                "prompt": prompt,
                "temperature": float(temperature),
                "metadata": metadata,
            },
        )
        cached = self.cache.get(key)
        if isinstance(cached, str):
            return cached
        if self.mode == "heuristic":
            return None
        text = self.client.generate(prompt, temperature=temperature)
        if text is None:
            if self.mode == "llm":
                raise RuntimeError(f"LLM call failed for {level} metadata={metadata}")
            return None
        self.cache.put(key, text, {"level": level, **metadata})
        return text

    def _cached_distribution(
        self,
        *,
        prompt: str,
        metadata: Dict[str, Any],
        sample_id: int,
    ) -> Optional[np.ndarray]:
        key = self.cache.make_key(
            level="regime",
            identity={
                "model": self.cfg.model,
                "prompt": prompt,
                "temperature": float(self.cfg.regime_temperature),
                "sample_id": int(sample_id),
                "metadata": metadata,
            },
        )
        cached = self.cache.get(key)
        q_cached = parse_regime_distribution(cached)
        if q_cached is not None:
            return q_cached
        if self.mode == "heuristic":
            return None
        text = self.client.generate(prompt, temperature=self.cfg.regime_temperature)
        q = parse_regime_distribution(text)
        if q is None:
            if self.mode == "llm":
                raise RuntimeError(f"LLM regime parser failed for metadata={metadata}; response={text!r}")
            return None
        self.cache.put(
            key,
            {"Bull": float(q[0]), "Bear": float(q[1]), "Neutral": float(q[2]), "raw_response": text},
            {"level": "regime", "sample_id": int(sample_id), **metadata},
        )
        return q

    def build_prior_for_day(
        self,
        env: Any,
        day: int,
        *,
        news_index: Optional[TitleNewsIndex] = None,
    ) -> Dict[str, Any]:
        day = int(day)
        absolute_day = int(getattr(env, "beg_idx", 0)) + day
        date_label = self.date_label(env, day, news_index)
        news_payload = self.news_payload(env, day, news_index)
        market_payload = self.market_feature_payload(env, day)
        metadata = {
            "market": str(getattr(env, "market", "")),
            "absolute_day": int(absolute_day),
            "day": int(day),
            "date": date_label,
        }

        event_prompt = self.NEWS_SUMMARY_PROMPT.format(
            date_label=date_label,
            news_payload=json.dumps(news_payload, ensure_ascii=False, sort_keys=True),
        )
        indicator_prompt = self.INDICATOR_SUMMARY_PROMPT.format(
            date_label=date_label,
            market_features=json.dumps(market_payload, ensure_ascii=False, sort_keys=True),
        )
        event_summary = self._cached_text(
            level="event_summary",
            prompt=event_prompt,
            metadata=metadata,
            temperature=self.cfg.summary_temperature,
        )
        indicator_summary = self._cached_text(
            level="indicator_summary",
            prompt=indicator_prompt,
            metadata=metadata,
            temperature=self.cfg.summary_temperature,
        )

        llm_samples: List[np.ndarray] = []
        source = "llm"
        if event_summary is not None and indicator_summary is not None:
            regime_prompt = self.REGIME_PROMPT.format(
                event_summary=event_summary,
                indicator_summary=indicator_summary,
            )
            for sample_id in range(max(int(self.cfg.samples), 1)):
                q_sample = self._cached_distribution(
                    prompt=regime_prompt,
                    metadata=metadata,
                    sample_id=sample_id,
                )
                if q_sample is not None:
                    llm_samples.append(q_sample)

        if llm_samples:
            q = sanitize_simplex(np.mean(np.asarray(llm_samples, dtype=np.float64), axis=0), len(REGIME_NAMES), 2)
        else:
            q_num = heuristic_regime_weights(compute_market_features(env, day))
            sector_titles = []
            for titles in news_payload.get("sectors", {}).values():
                sector_titles.extend(titles)
            sentiment = news_sentiment_score(sector_titles)
            news_logits = np.asarray([0.35 * sentiment, -0.35 * sentiment, -0.12 * abs(sentiment)], dtype=np.float64)
            logits = np.log(np.clip(q_num, 1e-8, 1.0)) + news_logits
            logits = logits - float(np.max(logits))
            exp = np.exp(logits)
            q = sanitize_simplex(exp / (float(exp.sum()) + 1e-12), len(REGIME_NAMES), 2)
            source = "heuristic"
            if self.mode == "llm":
                raise RuntimeError(f"missing valid LLM regime samples for {metadata}")

        return {
            "absolute_day": int(absolute_day),
            "day": int(day),
            "date": date_label,
            "q_bull": float(q[0]),
            "q_bear": float(q[1]),
            "q_neutral": float(q[2]),
            "confidence": float(normalized_entropy_confidence(q)),
            "source": source,
            "num_samples": int(len(llm_samples)),
        }


def write_prior_rows(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "absolute_day",
        "day",
        "date",
        "q_bull",
        "q_bear",
        "q_neutral",
        "confidence",
        "source",
        "num_samples",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return str(out_path)


__all__ = [
    "LLMRegimePriorBuilder",
    "LLMRegimePriorConfig",
    "LocalRegimeLLMClient",
    "PersistentLLMPriorCache",
    "normalized_entropy_confidence",
    "parse_regime_distribution",
    "write_prior_rows",
]
