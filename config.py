"""Central configuration for the RMT signals project."""
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Universe: S&P-style basket, swap for your own tickers
    tickers: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
        "JPM", "BAC", "GS", "MS", "C",
        "XOM", "CVX", "COP",
        "JNJ", "PFE", "UNH", "MRK",
        "PG", "KO", "PEP", "WMT",
        "DIS", "NFLX", "CMCSA",
        "CAT", "BA", "HON",
        "V", "MA"
    ])

    start_date: str = "2018-01-01"
    end_date: str = None          # None -> today
    price_field: str = "Close"    # yfinance auto_adjust=True gives adjusted Close

    # Rolling window for correlation estimation (trading days)
    lookback: int = 252

    # RMT
    mp_confidence: float = 1.0    # scale factor on MP upper edge (1.0 = theoretical)

    # Denoising method: "clip" (Laloux constant-residual) or "shrink" (target shrinkage)
    denoise_method: str = "clip"
    shrinkage_target: float = 0.5  # only used if denoise_method == "shrink"

    # Signal generation
    n_market_factors: int = 1     # eigenvectors treated as "market", excluded from stat-arb residual
    zscore_window: int = 20       # window for z-scoring residual cumulative returns
    entry_z: float = 1.25
    exit_z: float = 0.25

    # Backtest
    holding_cost_bps: float = 1.0  # per-side transaction cost in bps
