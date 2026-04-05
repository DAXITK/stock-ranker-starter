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


def test_purged_date_split():
    training, test, _ = temporal_split(build_features(synthetic_prices()))
    assert training["label_end"].max() < test["date"].min()
    assert not set(training["date"]) & set(test["date"])
    assert np.isfinite(training[FEATURES].to_numpy()).all()


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("model")
    data = root / "prices.csv"
    synthetic_prices().to_csv(data, index=False)
    out = root / "artifacts"
    train(data, out, "synthetic")
    return out


def test_api(artifacts):
    with TestClient(create_app(artifacts)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/model").json()["data_source"] == "synthetic"
        response = client.post("/predict", json={"tickers": ["aapl", "MSFT", "AAPL"]})
        assert response.status_code == 200
        rows = response.json()["rankings"]
        assert len(rows) == 2
        assert rows[0]["predicted_return"] >= rows[1]["predicted_return"]
        assert client.post("/predict", json={"tickers": []}).status_code == 422
        assert client.post("/predict", json={"tickers": ["INVALID"]}).status_code == 422
        assert client.get("/metrics").status_code == 200


def test_tampered_artifact_rejected(artifacts, tmp_path):
    import shutil
    copy = tmp_path / "tampered"
    shutil.copytree(artifacts, copy)
    with (copy / "model.json").open("a") as file:
        file.write(" ")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        with TestClient(create_app(copy)):
            pass
