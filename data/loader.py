"""Data acquisition via yfinance."""
import pandas as pd
import yfinance as yf
from config import Config


def load_prices(cfg: Config) -> pd.DataFrame:
    """Download adjusted close prices for the configured universe.

    Returns
    -------
    pd.DataFrame with DatetimeIndex, one column per ticker.
    """
    raw = yf.download(
        tickers=cfg.tickers,
        start=cfg.start_date,
        end=cfg.end_date,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # yfinance returns a MultiIndex column frame when multiple tickers requested
    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.concat(
            {t: raw[t][cfg.price_field] for t in cfg.tickers if t in raw.columns.get_level_values(0)},
            axis=1,
        )
    else:
        prices = raw[[cfg.price_field]].rename(columns={cfg.price_field: cfg.tickers[0]})

    prices = prices.dropna(axis=1, how="all")   # drop tickers with no data
    prices = prices.ffill().dropna()            # forward-fill holidays, drop leading NaNs
    return prices


def compute_returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    if log:
        rets = (prices / prices.shift(1)).apply(lambda x: x.map(lambda v: v)).pipe(
            lambda df: df.apply(lambda col: col)
        )
        import numpy as np
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    return rets.dropna()
