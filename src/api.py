import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from xgboost import XGBRegressor

from src.config import FEATURES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("stock-ranker")
REQUESTS = Counter("stock_api_requests", "HTTP requests", ["route", "status"])
LATENCY = Histogram("stock_api_latency_seconds", "HTTP request latency", ["route"])


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tickers: list[str] = Field(min_length=1, max_length=50)


def create_app(artifact_dir: Path | None = None) -> FastAPI:
    directory = artifact_dir or Path(os.getenv("ARTIFACT_DIR", "artifacts/demo"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata["features"] != FEATURES:
            raise RuntimeError("Feature schema mismatch")
        for name, key in [("model.json", "model_sha256"),
                          ("latest_features.csv", "snapshot_sha256")]:
            digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            if digest != metadata[key]:
                raise RuntimeError(f"Artifact hash mismatch: {name}")
        model = XGBRegressor(n_jobs=1)
        model.load_model(directory / "model.json")
        snapshot = pd.read_csv(directory / "latest_features.csv")
        scores = model.predict(snapshot[FEATURES])
        app.state.predictions = dict(zip(snapshot["ticker"], map(float, scores)))
        app.state.metadata = metadata
        yield

    app = FastAPI(title="Stock Factor Ranking API", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            duration = time.perf_counter() - started
            REQUESTS.labels(path, str(status)).inc()
            LATENCY.labels(path).observe(duration)
            logger.info(json.dumps({"route": path, "status": status,
                                    "latency_ms": round(duration * 1000, 2)}))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        if not hasattr(app.state, "predictions"):
            raise HTTPException(503, "Model not loaded")
        return {"status": "ready", "model_version": app.state.metadata["model_version"]}

    @app.get("/model")
    def model_info():
        return {**app.state.metadata, "supported_tickers": sorted(app.state.predictions)}

    @app.post("/predict")
    def predict(body: PredictionRequest):
        tickers = list(dict.fromkeys(t.strip().upper() for t in body.tickers))
        missing = sorted(set(tickers) - set(app.state.predictions))
        if missing:
            raise HTTPException(422, {"unsupported_tickers": missing})
        ranked = sorted(tickers, key=lambda t: app.state.predictions[t], reverse=True)
        return {
            "model_version": app.state.metadata["model_version"],
            "as_of": app.state.metadata["snapshot_as_of"],
            "data_source": app.state.metadata["data_source"],
            "horizon_trading_days": app.state.metadata["horizon_trading_days"],
            "rankings": [{"ticker": ticker, "rank": i + 1,
                          "predicted_return": app.state.predictions[ticker]}
                         for i, ticker in enumerate(ranked)],
            "disclaimer": "Dated snapshot; educational demo; not investment advice.",
        }

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})

    return app


app = create_app()
