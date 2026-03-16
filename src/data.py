import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import DATA_PATH, DEFAULT_TICKERS


def synthetic_prices() -> pd.DataFrame:
    """Deterministic invented prices for tests and publicly shareable demos."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=900)
    frames = []
    for ticker in DEFAULT_TICKERS:
        returns = rng.normal(0.0003, 0.015, len(dates))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": ticker,
            "close": 100 * np.exp(np.cumsum(returns)),
            "volume": rng.integers(100_000, 2_000_000, len(dates)),
        }))
    return pd.concat(frames, ignore_index=True)


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        frame = yf.download(
            ticker, start=start, end=end, auto_adjust=True,
            multi_level_index=False, progress=False, threads=False,
        )
        if frame is None or frame.empty:
            raise RuntimeError(f"No data for {ticker}; retry later or use --synthetic.")
        frame = frame.reset_index().rename(columns={
            "Date": "date", "Close": "close", "Volume": "volume",
        })
        frame["ticker"] = ticker
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        frames.append(frame[["date", "ticker", "close", "volume"]])
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-01-01", help="Exclusive end date")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--out", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    frame = synthetic_prices() if args.synthetic else download_prices(
        args.tickers, args.start, args.end
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Saved {len(frame):,} rows to {args.out}")


if __name__ == "__main__":
    main()
