import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.config import FEATURES
from src.data import synthetic_prices
from src.features import build_features, temporal_split
from src.train import train


def test_no_future_feature_leakage():
    prices = synthetic_prices()
    cutoff = pd.Timestamp("2022-01-01")
    original = build_features(prices)
    changed = prices.copy()
    changed.loc[changed["date"] > cutoff, "close"] *= 3
    altered = build_features(changed)
    pd.testing.assert_frame_equal(
        original.loc[original["date"] <= cutoff, FEATURES],
        altered.loc[altered["date"] <= cutoff, FEATURES],
    )

