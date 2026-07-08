"""Minimal vectorized backtest for the RMT residual mean-reversion signal."""
import numpy as np
import pandas as pd
from config import Config


def run_backtest(returns: pd.DataFrame, positions: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Equal-weight the active positions each day, apply next-day returns,
    and charge transaction costs on position changes."""
    positions = positions.shift(1).fillna(0.0)   # trade on next bar to avoid look-ahead

    n_active = positions.abs().sum(axis=1).replace(0, np.nan)
    weights = positions.div(n_active, axis=0).fillna(0.0)

    gross_pnl = (weights * returns).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * (cfg.holding_cost_bps / 1e4)

    net_pnl = gross_pnl - costs
    equity = (1 + net_pnl).cumprod()

    result = pd.DataFrame({
        "gross_pnl": gross_pnl,
        "costs": costs,
        "net_pnl": net_pnl,
        "equity": equity,
    })
    return result


def performance_summary(result: pd.DataFrame, freq: int = 252) -> dict:
    r = result["net_pnl"]
    ann_return = r.mean() * freq
    ann_vol = r.std() * np.sqrt(freq)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    max_dd = (result["equity"] / result["equity"].cummax() - 1).min()
    return {
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }
