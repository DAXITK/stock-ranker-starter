import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from src.config import DATA_PATH, FEATURES, HORIZON
from src.features import build_features, temporal_split


def train(data: Path, out: Path, source: str) -> dict:
    if out.exists() and any(p.name != ".gitkeep" for p in out.iterdir()):
        raise ValueError("Use a new, empty artifact directory for each model version")
    prices = pd.read_csv(data)
    frame = build_features(prices)
    training, test, cutoff = temporal_split(frame)
    model = XGBRegressor(
        n_estimators=120, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", random_state=42, n_jobs=1,
    )
    model.fit(training[FEATURES], training["target"])
    predicted = model.predict(test[FEATURES])
    evaluation = test[["date", "ticker", "target"]].copy()
    evaluation["prediction"] = predicted
    correlations = []
    for _, day in evaluation.groupby("date"):
        if day["prediction"].nunique() > 1 and day["target"].nunique() > 1:
            correlations.append(day["prediction"].rank().corr(day["target"].rank()))
    metrics = {
        "test_mae": float(mean_absolute_error(test["target"], predicted)),
        "zero_return_baseline_mae": float(mean_absolute_error(
            test["target"], np.zeros(len(test))
        )),
        "daily_rank_correlation": float(np.mean(correlations)) if correlations else None,
    }
    latest_date = frame["date"].max()
    latest = frame[frame["date"] == latest_date][["ticker", *FEATURES]].copy()
    if latest.empty:
        raise ValueError("No latest feature rows")
    out.mkdir(parents=True, exist_ok=True)
    model.save_model(out / "model.json")
    latest.to_csv(out / "latest_features.csv", index=False)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    metadata = {
        "model_version": f"{stamp}-{digest[:8]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_source": source, "data_sha256": digest,
        "model_sha256": hashlib.sha256((out / "model.json").read_bytes()).hexdigest(),
        "snapshot_sha256": hashlib.sha256((out / "latest_features.csv").read_bytes()).hexdigest(),
        "features": FEATURES, "horizon_trading_days": HORIZON,
        "snapshot_as_of": latest_date.date().isoformat(),
        "train_start": training["date"].min().date().isoformat(),
        "train_end": training["date"].max().date().isoformat(),
        "test_start": cutoff.date().isoformat(),
        "test_end": test["date"].max().date().isoformat(),
        "train_rows": len(training), "test_rows": len(test), "metrics": metrics,
        "versions": {"python": platform.python_version(), "xgboost": xgboost.__version__,
                     "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "limitations": [
            "Fixed-universe survivorship bias; not an investment recommendation.",
            "Single chronological holdout, not nested walk-forward validation.",
            "Overlapping five-day labels; observations are not independent.",
            "Predictions use a dated snapshot, not live market data.",
            "Features assume the trading session has closed.",
            "Adjusted historical prices can be revised; no point-in-time fundamentals.",
        ],
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
    print(json.dumps(metadata, indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", choices=["synthetic", "yahoo-personal"], required=True)
    args = parser.parse_args()
    train(args.data, args.out, args.source)


if __name__ == "__main__":
    main()
