"""Transparent spectral-response and sensitivity models."""

from __future__ import annotations

import numpy as np


def fwhm_um(wavelength_um: np.ndarray) -> np.ndarray:
    """Piecewise baseline resolution for VNIR, MIR, and FIR concepts.

    The model uses M3-like 0.01 um VNIR resolution, an explicitly illustrative
    0.05--0.40 um MIR transition, and the 0.6--1.2 um range described for the
    20--40 um MIRORES concept. It is a configuration envelope, not one optical
    design spanning all regimes.
    """
    wavelength = np.asarray(wavelength_um, dtype=float)
    result = np.empty_like(wavelength)
    vnir = wavelength <= 3.0
    mir = (wavelength > 3.0) & (wavelength < 20.0)
    fir = wavelength >= 20.0
    result[vnir] = 0.01
    result[mir] = 0.05 + (wavelength[mir] - 3.0) * (0.40 - 0.05) / 17.0
    result[fir] = 0.60 + (wavelength[fir] - 20.0) * (1.20 - 0.60) / 20.0
    return result


def region_snr(wavelength_um: np.ndarray, profile: dict[str, float]) -> np.ndarray:
    wavelength = np.asarray(wavelength_um, dtype=float)
    return np.where(
        wavelength <= 3.0,
        profile["vnir_snr"],
        np.where(wavelength < 20.0, profile["mir_snr"], profile["fir_snr"]),
    ).astype(float)


def noise_sigma(
    response_matrix: np.ndarray,
    wavelength_um: np.ndarray,
    profile: dict[str, float],
) -> np.ndarray:
    """Return signal-scaled noise for a stated SNR sensitivity profile."""
    scale = np.maximum(np.median(np.abs(response_matrix), axis=1), 0.05)
    return scale / region_snr(wavelength_um, profile)


def valid_candidate_centers(
    low_um: float,
    high_um: float,
    step_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create centers whose four-sigma Gaussian support remains measured."""
    raw = np.arange(low_um, high_um + step_um / 2, step_um)
    widths = fwhm_um(raw)
    margin = 1.70 * widths  # approximately four Gaussian standard deviations
    valid = (raw - margin >= low_um) & (raw + margin <= high_um)
    return raw[valid], widths[valid]
