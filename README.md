# Stock Factor Ranking API

An educational production-oriented ML service: local data preparation and XGBoost
training, purged chronological evaluation, versioned JSON artifacts, FastAPI
serving, Prometheus instrumentation, tests, Docker, and GitHub Actions CI.

This is a supervised regressor whose scores are sorted to produce rankings, not
an XGBoost learning-to-rank model. It uses price/volume technical features, NOT
point-in-time value or quality fundamentals. Those are a later extension.

## 1. Install locally

Install Python 3.11 and Git. Use a virtual environment. Docker is only needed at
step 6. Commands assume you are in this repository's root.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Windows PowerShell activation:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

Pinned versions are an initial reproducibility baseline, not a promise of latest
security patches. Review/update dependencies and run CI before public releases.

## 2. Create the first dataset

Start offline with deterministic synthetic data:

```bash
python -m src.data --synthetic --out data/demo.csv
```

This uses no network, no external tokens and no model API. Prices are invented,
so the resulting metrics say nothing about real market performance.

For your private research run, download actual historical prices separately:

```bash
python -m src.data --start 2020-01-01 --end 2026-01-01 --out data/prices.csv
```

The end date is exclusive. yfinance access is unofficial, may fail or be rate
limited, and is intended for personal research/educational use. Check Yahoo's
terms before sharing market data, derived snapshots, or artifacts. Do not commit
raw Yahoo data or publish a Yahoo-derived demo without checking permitted use.

## 3. Train and evaluate

```bash
python -m src.train --data data/demo.csv --out artifacts/demo --source synthetic
```

The directory must be empty or contain only the .gitkeep placeholder. For later
runs, use a new versioned output directory rather than overwriting a model.
Do not delete existing real artifacts accidentally.

Training produces model.json, metadata.json and latest_features.csv. It predicts
five-trading-observation forward returns using features available after a daily
session closes. Predictions are decimal returns: 0.01 means 1%, not a probability.

The first 80% of labeled dates are training candidates and the final 20% form an
untouched test period. Training labels reaching the test period are purged. All
tickers for a date stay together. No random row split. Hyperparameters are fixed;
do not repeatedly tune against this holdout. Use rolling validation within the
training period before introducing hyperparameter tuning.

The saved model remains the training-period model evaluated on the holdout; it
is NOT refitted on test labels. This makes metrics attributable to the saved
artifact. A separately evaluated and versioned final-fit workflow is future work.

Read metadata.json. Compare test_mae to zero_return_baseline_mae. Lower MAE is
better; positive daily_rank_correlation is better than negative. Poor results are
valid results. No financial-performance guarantee or fabricated accuracy claim.

For private real-data training:

```bash
python -m src.train --data data/prices.csv --out artifacts/private-v1 --source yahoo-personal
```

The --source argument records your declaration; it cannot prove dataset origin.

## 4. Test the pipeline

```bash
ruff check src tests
pytest -q
```

Tests use synthetic data and train a small temporary model. They check future
price perturbations cannot change past features, training labels do not cross
the test boundary, API validation works, and modified artifacts fail checksums.
Checksums detect accidental modification, not malicious replacement of both the
artifact and metadata. Only deploy artifacts from trusted builds.

## 5. Run the API

```bash
uvicorn src.api:app --host 127.0.0.1 --port 7860
```

Open http://localhost:7860/docs in your browser. Use POST /predict:

```json
{"tickers": ["AAPL", "MSFT", "NVDA"]}
```

Endpoints:
- GET /health: process liveness.
- GET /ready: model loaded and ready.
- GET /model: version, provenance, evaluation and supported tickers.
- POST /predict: rank requested supported tickers using the saved snapshot.
- GET /metrics: process-local Prometheus HTTP counts/latency.

On Linux/macOS, choose a different artifact version with:

```bash
ARTIFACT_DIR=artifacts/private-v1 uvicorn src.api:app --port 7860
```

On PowerShell:

```powershell
$env:ARTIFACT_DIR = "artifacts/private-v1"
uvicorn src.api:app --port 7860
```

Predictions use the exact dated snapshot stated in responses. The API does not
fetch prices on requests, does not automatically refresh data, and does not
estimate delayed real-world accuracy or feature drift. Metrics reset on restart.

## 6. Deploy the container locally

First create the synthetic artifacts in artifacts/demo. Then:

```bash
docker compose up --build -d
docker compose logs -f api
```

Open http://localhost:7860/docs. Stop with:

```bash
docker compose down
```

The model is copied into the image. No training or data download occurs during
container build or startup. One process avoids duplicating model memory. The
container uses a non-root user. Missing/corrupt artifacts cause startup failure.

## 7. Put the project in GitHub

Create an empty public GitHub repository. Then run these manually, replacing the
example remote with your actual repository URL:

```bash
git init
git add .
git commit -m "Build tested stock-ranking ML service"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/stock-ranker.git
git push -u origin main
```

Only synthetic demo artifacts are allowed by the provided .gitignore; inspect
git status and ensure data_source is synthetic in artifacts/demo/metadata.json.
Do not commit tokens or private datasets. Choose an open-source license before
sharing; no license was selected on your behalf.

The workflow tests the code, trains a throwaway synthetic model, uploads it as a
CI artifact, builds the Docker image and checks readiness. These are CI artifacts,
not automatically promoted production models.

## 8. Optional free public deployment

Use Render's dashboard to connect this GitHub repository. Create a Web Service:
- Runtime: Docker.
- Instance type: Free.
- Dockerfile: ./Dockerfile.
- Health check path: /ready.
- Auto-deploy: After CI Checks Pass, if available in your dashboard.
- Payment method: do not add one for this zero-budget demo.

If an option is paid or requires a payment method, stop and keep the local Docker
deployment instead. Do not choose a paid cron job, database, or persistent disk.
The app binds to Render's PORT environment variable. No API keys required.

After deployment, open your assigned service URL plus /docs and /model. Confirm
source, version and snapshot date match your local artifact. Test POST /predict.
A deployment restart uses artifacts built into the image, not ephemeral files.

Render's free tier currently sleeps after 15 idle minutes and shares 750 monthly
instance hours per workspace. It has bandwidth/build limits; without a payment
method the documented behavior is suspension instead of overage billing.
Verify current pricing before deploying. This is a demo deployment, not an SLA.

## 9. Next engineering increments

1. Expand chronological validation to rolling windows; purge labels at every boundary.
2. Add local MLflow run tracking and a candidate/champion promotion gate.
3. Build a batch refresh command that downloads prices once and creates a new,
   immutable model/snapshot bundle; retain the previous bundle for rollback.
4. Add feature-distribution drift checks against stored training summaries.
5. Store prediction events locally and evaluate outcomes only once labels mature;
   Prometheus HTTP metrics are not model-quality monitoring.
6. Add a small dashboard, authentication/rate limits if exposing expensive work,
   and dependency vulnerability scans.

Do not add Airflow, Kubernetes, databases or paid infrastructure before the basic
train-test-serve-deploy loop works. No component in this scaffold calls an LLM.

## Known limitations

Fixed current ticker universe introduces survivorship/selection bias. Five-day
labels overlap. Historical adjusted data may change and is not a point-in-time
investment database. No costs/slippage/backtest claims. No automatic retraining,
model-quality monitoring, durable metrics, or promotion/rollback automation yet.
This project demonstrates production-oriented engineering, not production
investment readiness.
