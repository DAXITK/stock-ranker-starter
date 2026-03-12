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

