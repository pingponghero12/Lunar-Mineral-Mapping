"""Simple Monte Carlo assessment for selected band configurations."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def project_simplex(vector: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the non-negative unit simplex."""
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered)
    rho_candidates = ordered - (cumulative - 1.0) / np.arange(1, vector.size + 1) > 0
    rho = np.nonzero(rho_candidates)[0][-1]
    theta = (cumulative[rho] - 1.0) / (rho + 1)
    return np.maximum(vector - theta, 0.0)


def monte_carlo_unmixing(
    model_response: np.ndarray,
    actual_response: np.ndarray,
    sigma: np.ndarray,
    alpha: np.ndarray,
    trials: int,
    continuum_sigma: float,
    seed: int,
) -> dict[str, float | list[float]]:
    """Evaluate NNLS-plus-simplex estimates for known synthetic mixtures."""
    rng = np.random.default_rng(seed)
    truth = rng.dirichlet(alpha, size=trials)
    axis = np.linspace(-1.0, 1.0, model_response.shape[0])
    estimates = np.empty_like(truth)
    for row in range(trials):
        slopes = rng.normal(0.0, continuum_sigma, size=model_response.shape[1])
        perturbed = actual_response * (1.0 + axis[:, None] * slopes[None, :])
        measurement = perturbed @ truth[row] + rng.normal(0.0, sigma)
        coefficients, _ = nnls(model_response, measurement)
        estimates[row] = project_simplex(coefficients)
    error = estimates - truth
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "maximum_absolute_error": float(absolute.max()),
        "component_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "trials": int(trials),
    }
