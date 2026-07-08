"""Entry point: end-to-end RMT denoising + signal pipeline."""
from config import Config
from data.loader import load_prices, compute_returns
from rmt.denoise import denoise_correlation, eigen_decompose
from signals.generator import generate_signals
from backtest.engine import run_backtest, performance_summary


def main():
    cfg = Config()

    print("Downloading price data...")
    prices = load_prices(cfg)
    returns = compute_returns(prices)
    print(f"Universe: {returns.shape[1]} tickers, {returns.shape[0]} trading days")

    # One-shot RMT diagnostic on the full sample corr matrix
    corr = returns.corr()
    T, N = returns.shape
    denoised_corr, info = denoise_correlation(corr, T, method=cfg.denoise_method)
    print(f"MP noise band: [{info['lambda_minus']:.3f}, {info['lambda_plus']:.3f}]")
    print(f"Signal eigenvalues detected: {info['n_signal_factors']} out of {N}")

    print("Generating rolling RMT residual mean-reversion signals...")
    positions = generate_signals(returns, cfg)

    print("Running backtest...")
    result = run_backtest(returns, positions, cfg)
    stats = performance_summary(result)

    print("\n--- Performance ---")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")

    return result, stats, info


if __name__ == "__main__":
    main()
