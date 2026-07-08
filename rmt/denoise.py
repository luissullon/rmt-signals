"""Eigenvalue-based denoising of correlation matrices."""
import numpy as np
import pandas as pd
from rmt.marchenko_pastur import mp_bounds


def eigen_decompose(corr: pd.DataFrame):
    """Symmetric eigendecomposition, sorted descending by eigenvalue."""
    eigvals, eigvecs = np.linalg.eigh(corr.values)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def denoise_correlation(
    corr: pd.DataFrame,
    T: int,
    method: str = "clip",
    confidence: float = 1.0,
    shrinkage_target: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Denoise a correlation matrix using RMT.

    method="clip": eigenvalues inside the MP noise band are replaced by their
        average (Laloux et al. 'constant residual eigenvalue' method), which
        preserves the trace (=N) of the correlation matrix.
    method="shrink": noise eigenvalues are shrunk toward shrinkage_target * mean,
        a softer alternative when the clip method overfits denoised factors.
    """
    N = corr.shape[0]
    eigvals, eigvecs = eigen_decompose(corr)
    lam_minus, lam_plus = mp_bounds(T, N, confidence=confidence)

    is_signal = eigvals > lam_plus
    n_signal = int(is_signal.sum())

    denoised_eigvals = eigvals.copy()
    noise_mask = ~is_signal

    if noise_mask.any():
        if method == "clip":
            avg_noise = eigvals[noise_mask].mean()
            denoised_eigvals[noise_mask] = avg_noise
        elif method == "shrink":
            mean_noise = eigvals[noise_mask].mean()
            denoised_eigvals[noise_mask] = (
                shrinkage_target * eigvals[noise_mask] + (1 - shrinkage_target) * mean_noise
            )
        else:
            raise ValueError(f"Unknown denoise method: {method}")

    # Reconstruct: C_denoised = V * diag(lambda) * V^T, then renormalize to unit diagonal
    C = eigvecs @ np.diag(denoised_eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)

    denoised = pd.DataFrame(C, index=corr.index, columns=corr.columns)

    info = {
        "eigvals_raw": eigvals,
        "eigvecs": eigvecs,
        "lambda_minus": lam_minus,
        "lambda_plus": lam_plus,
        "n_signal_factors": n_signal,
    }
    return denoised, info
