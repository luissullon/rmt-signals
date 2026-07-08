"""Signal generation from RMT-denoised structure.

Strategy: strip the top `n_market_factors` eigenvectors (systematic/market
risk) from returns to get idiosyncratic residuals, then trade mean-reversion
in the cumulative residual (a classic stat-arb / PCA pairs-style approach),
but only within the signal subspace confirmed by RMT (i.e., only eigenvectors
above the MP upper edge are trusted as real common factors).
"""
import numpy as np
import pandas as pd
from config import Config
from rmt.denoise import eigen_decompose, denoise_correlation


def compute_residual_returns(
    returns: pd.DataFrame, eigvecs: np.ndarray, n_factors: int
) -> pd.DataFrame:
    """Project returns onto top n_factors eigenvectors and subtract that
    systematic component, leaving idiosyncratic residual returns."""
    F = eigvecs[:, :n_factors]                     # N x k factor loadings
    std = returns.std()
    standardized = (returns - returns.mean()) / std
    factor_returns = standardized.values @ F        # T x k
    systematic = factor_returns @ F.T                # T x N, back in standardized space
    residual_std = standardized.values - systematic
    residual = residual_std * std.values             # back to return units
    return pd.DataFrame(residual, index=returns.index, columns=returns.columns)


def zscore(series: pd.Series, window: int) -> pd.Series:
    roll_mean = series.rolling(window).mean()
    roll_std = series.rolling(window).std()
    return (series - roll_mean) / roll_std


def generate_signals(returns: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Rolling-window RMT denoise + residual mean-reversion signal.

    Returns a DataFrame of positions in {-1, 0, +1} per ticker per day
    (long residual when very negative/oversold, short when very positive).
    """
    n = len(returns)
    tickers = returns.columns
    positions = pd.DataFrame(0.0, index=returns.index, columns=tickers)

    for end in range(cfg.lookback, n):
        window = returns.iloc[end - cfg.lookback : end]
        corr = window.corr()
        T = cfg.lookback

        denoised_corr, info = denoise_correlation(
            corr, T, method=cfg.denoise_method,
            confidence=cfg.mp_confidence,
            shrinkage_target=cfg.shrinkage_target,
        )
        eigvals, eigvecs = eigen_decompose(denoised_corr)

        n_factors = max(cfg.n_market_factors, 1)
        residual = compute_residual_returns(window, eigvecs, n_factors)
        cum_residual = residual.cumsum()

        z = cum_residual.apply(lambda col: zscore(col, cfg.zscore_window)).iloc[-1]

        date = returns.index[end]
        for ticker in tickers:
            zi = z.get(ticker, np.nan)
            if pd.isna(zi):
                continue
            prev = positions.iloc[end - 1][ticker] if end > 0 else 0.0
            if zi > cfg.entry_z:
                positions.loc[date, ticker] = -1.0       # overextended up -> fade it
            elif zi < -cfg.entry_z:
                positions.loc[date, ticker] = 1.0        # overextended down -> buy the dip
            elif abs(zi) < cfg.exit_z:
                positions.loc[date, ticker] = 0.0
            else:
                positions.loc[date, ticker] = prev        # hold

    return positions
