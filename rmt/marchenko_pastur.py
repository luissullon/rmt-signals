"""Marchenko-Pastur theoretical spectrum utilities."""
import numpy as np


def mp_bounds(T: int, N: int, sigma2: float = 1.0, confidence: float = 1.0) -> tuple[float, float]:
    """Theoretical min/max eigenvalues of a NxN correlation matrix built from
    T iid samples of N variables with variance sigma2, per Marchenko-Pastur.

    Q = T / N must be >= 1 for the standard MP law.
    """
    Q = T / N
    lambda_plus = sigma2 * (1 + 1 / np.sqrt(Q)) ** 2
    lambda_minus = sigma2 * (1 - 1 / np.sqrt(Q)) ** 2
    return lambda_minus * confidence, lambda_plus * confidence


def mp_pdf(x: np.ndarray, T: int, N: int, sigma2: float = 1.0) -> np.ndarray:
    """Theoretical MP density, useful for diagnostic plots."""
    Q = T / N
    lam_minus, lam_plus = mp_bounds(T, N, sigma2)
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    mask = (x > lam_minus) & (x < lam_plus)
    y[mask] = (Q / (2 * np.pi * sigma2 * x[mask])) * np.sqrt(
        (lam_plus - x[mask]) * (x[mask] - lam_minus)
    )
    return y
