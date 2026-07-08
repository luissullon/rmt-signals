# rmt-signals

End-to-end pipeline that uses Random Matrix Theory to denoise equity
correlation matrices and extract a residual mean-reversion trading signal.

## Pipeline
1. **data/loader.py** — pull adjusted daily prices via `yfinance`, compute log returns.
2. **rmt/marchenko_pastur.py** — theoretical MP eigenvalue bounds for a TxN random matrix.
3. **rmt/denoise.py** — eigendecompose the empirical correlation matrix, replace
   noise-band eigenvalues (those below the MP upper edge) with their average,
   reconstruct a denoised correlation matrix.
4. **signals/generator.py** — on a rolling window, use the denoised eigenvectors
   to strip out systematic (market) factors, leaving idiosyncratic residuals;
   z-score the cumulative residual and trade mean-reversion in it.
5. **backtest/engine.py** — equal-weight active names, apply next-day returns,
   charge simple transaction costs, report Sharpe / drawdown.

## Usage
\`\`\`bash
pip install -r requirements.txt
python main.py
\`\`\`

## Notes
- `Q = T/N` should stay >= 1 for the MP law to be well defined — keep the
  lookback window (`cfg.lookback`) larger than the number of tickers.
- `denoise_method="clip"` follows Laloux, Cizeau, Bouchaud & Potters (1999);
  `"shrink"` is a softer alternative if clipping proves too aggressive.
- This is a research scaffold, not investment advice — validate out-of-sample,
  check for survivorship bias in your ticker list, and stress-test transaction
  cost assumptions before trusting the Sharpe number.
