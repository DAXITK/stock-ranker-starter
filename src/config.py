from pathlib import Path

FEATURES = ["return_1d", "momentum_5d", "momentum_20d", "volatility_20d", "volume_ratio"]
HORIZON = 5
DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "WMT"]
DATA_PATH = Path("data/prices.csv")
