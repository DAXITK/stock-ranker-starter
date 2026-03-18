import numpy as np
import pandas as pd

from src.config import FEATURES, HORIZON


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "close", "volume"}
    if not required.issubset(prices.columns):
        raise ValueError(f"Required columns: {sorted(required)}")
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    if prices.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker rows")
    if prices[["close", "volume"]].isna().any().any():
        raise ValueError("Missing price/volume data")
    if not np.isfinite(prices[["close", "volume"]].to_numpy()).all():
        raise ValueError("Non-finite price/volume data")
    if (prices["close"] <= 0).any() or (prices["volume"] < 0).any():
        raise ValueError("Invalid prices or volumes")
    frames = []
    for _, group in prices.groupby("ticker"):
        group = group.sort_values("date").copy()
        close = group["close"]
        daily = close.pct_change(fill_method=None)
        group["return_1d"] = daily
        group["momentum_5d"] = close.pct_change(5, fill_method=None)
        group["momentum_20d"] = close.pct_change(20, fill_method=None)
        group["volatility_20d"] = daily.rolling(20).std()
        group["volume_ratio"] = group["volume"] / group["volume"].rolling(20).mean()
        group["target"] = close.shift(-HORIZON) / close - 1
        group["label_end"] = group["date"].shift(-HORIZON)
        frames.append(group)
    if not frames:
        raise ValueError("Empty prices")
    result = pd.concat(frames, ignore_index=True)
    result[FEATURES] = result[FEATURES].replace([np.inf, -np.inf], np.nan)
    return result.dropna(subset=FEATURES).sort_values(["date", "ticker"]).reset_index(drop=True)


def temporal_split(frame: pd.DataFrame):
    """Split unique dates, then purge labels extending into the test interval."""
    labeled = frame.dropna(subset=["target", "label_end"])
    dates = sorted(labeled["date"].unique())
    if len(dates) < 100:
        raise ValueError("Need at least 100 labeled dates")
    cutoff = pd.Timestamp(dates[int(len(dates) * 0.8)])
    train = labeled[(labeled["date"] < cutoff) & (labeled["label_end"] < cutoff)].copy()
    test = labeled[labeled["date"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Empty train/test split")
    return train, test, cutoff
