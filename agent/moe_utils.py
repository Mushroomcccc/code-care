from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def normalize_name(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _clean_title(value: Any) -> str:
    text = " ".join(str(value).strip().split())
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text[:50]


def _append_unique(dst: List[str], values: Iterable[str], max_items: int) -> None:
    seen = set(dst)
    for value in values:
        text = _clean_title(value)
        if not text or text in seen:
            continue
        dst.append(text)
        seen.add(text)
        if len(dst) >= int(max_items):
            return


@dataclass(frozen=True)
class TitleNewsIndex:
    """Title-only news lookup backed by *_stock_with_news.csv."""

    market: str
    csv_path: str
    dates: Tuple[str, ...]
    tickers: Tuple[str, ...]
    ticker_to_industry: Dict[str, str]
    titles_by_date_sector: Dict[Tuple[str, str], Tuple[str, ...]]
    titles_by_date_ticker: Dict[Tuple[str, str], Tuple[str, ...]]

    @classmethod
    def from_market(cls, market: str, data_dir: Optional[str] = None) -> "TitleNewsIndex":
        market = str(market).strip()
        if not market:
            raise ValueError("market must be a non-empty string")
        base_dir = Path(data_dir).expanduser().resolve() if data_dir else Path(__file__).resolve().parents[1] / "data"
        csv_path = base_dir / f"{market}_stock_with_news.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"title news csv not found: {csv_path}")

        dates: List[str] = []
        seen_dates = set()
        tickers: List[str] = []
        seen_tickers = set()
        ticker_to_industry: Dict[str, str] = {}
        by_sector_tmp: Dict[Tuple[str, str], List[str]] = {}
        by_ticker_tmp: Dict[Tuple[str, str], List[str]] = {}

        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            required = {"Date", "Ticker", "Industry", "title"}
            missing = sorted(required.difference(fieldnames))
            if missing:
                raise ValueError(f"{csv_path} is missing required columns: {missing}")

            for row in reader:
                date = str(row.get("Date", "")).strip()
                ticker = str(row.get("Ticker", "")).strip()
                industry = str(row.get("Industry", "")).strip()
                title = _clean_title(row.get("title", ""))
                if not date or not ticker:
                    continue
                if date not in seen_dates:
                    dates.append(date)
                    seen_dates.add(date)
                if ticker not in seen_tickers:
                    tickers.append(ticker)
                    seen_tickers.add(ticker)
                if ticker and industry:
                    ticker_to_industry.setdefault(ticker, industry)
                if not title:
                    continue

                sector_key = (date, normalize_name(industry))
                ticker_key = (date, ticker)
                by_sector_tmp.setdefault(sector_key, [])
                by_ticker_tmp.setdefault(ticker_key, [])
                _append_unique(by_sector_tmp[sector_key], [title], max_items=10_000)
                _append_unique(by_ticker_tmp[ticker_key], [title], max_items=10_000)

        return cls(
            market=market,
            csv_path=str(csv_path),
            dates=tuple(dates),
            tickers=tuple(tickers),
            ticker_to_industry=ticker_to_industry,
            titles_by_date_sector={key: tuple(values) for key, values in by_sector_tmp.items()},
            titles_by_date_ticker={key: tuple(values) for key, values in by_ticker_tmp.items()},
        )

    def _date_window(self, absolute_day: int, lookback_days: int) -> List[str]:
        if not self.dates:
            return []
        day = int(np.clip(int(absolute_day), 0, len(self.dates) - 1))
        start = max(0, day - max(int(lookback_days), 1) + 1)
        return list(self.dates[start : day + 1])

    def sector_titles(
        self,
        absolute_day: int,
        sector_name: Any,
        *,
        lookback_days: int = 1,
        max_titles: int = 8,
    ) -> List[str]:
        titles: List[str] = []
        sector_key = normalize_name(sector_name)
        for date in self._date_window(absolute_day, lookback_days):
            _append_unique(
                titles,
                self.titles_by_date_sector.get((date, sector_key), ()),
                max_items=max_titles,
            )
        return titles[: int(max_titles)]

    def stock_titles(
        self,
        absolute_day: int,
        ticker: Any,
        *,
        lookback_days: int = 1,
        max_titles: int = 4,
    ) -> List[str]:
        titles: List[str] = []
        ticker = str(ticker).strip()
        for date in self._date_window(absolute_day, lookback_days):
            _append_unique(
                titles,
                self.titles_by_date_ticker.get((date, ticker), ()),
                max_items=max_titles,
            )
        return titles[: int(max_titles)]

    def ticker_for_index(self, stock_index: int) -> str:
        i = int(stock_index)
        if 0 <= i < len(self.tickers):
            return self.tickers[i]
        return f"Stock-{i}"


def news_sentiment_score(titles: Sequence[str]) -> float:
    """Small deterministic fallback signal when no cached LLM prior is available."""

    if not titles:
        return 0.0
    positive = {
        "beat",
        "beats",
        "growth",
        "gain",
        "gains",
        "profit",
        "profits",
        "raises",
        "upgrade",
        "surge",
        "record",
        "strong",
        "wins",
    }
    negative = {
        "miss",
        "falls",
        "fall",
        "loss",
        "losses",
        "downgrade",
        "cuts",
        "cut",
        "risk",
        "probe",
        "weak",
        "slump",
        "lawsuit",
    }
    score = 0
    total = 0
    for title in titles:
        words = [word.strip(".,:;!?()[]{}'\"").lower() for word in str(title).split()]
        for word in words:
            if not word:
                continue
            total += 1
            if word in positive:
                score += 1
            elif word in negative:
                score -= 1
    if total <= 0:
        return 0.0
    return float(np.clip(score / math.sqrt(total), -2.0, 2.0))


__all__ = [
    "TitleNewsIndex",
    "news_sentiment_score",
    "normalize_name",
]
