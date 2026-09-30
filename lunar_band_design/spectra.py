"""RELAB spectrum loading with explicit support checks and no extrapolation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Spectrum:
    name: str
    path: Path
    wavelength_um: np.ndarray
    value: np.ndarray
    sha256: str

    @property
    def support(self) -> tuple[float, float]:
        return float(self.wavelength_um[0]), float(self.wavelength_um[-1])


def load_relab_spectrum(path: Path, name: str | None = None) -> Spectrum:
    """Load a two-column RELAB export, preserving only finite ordered samples."""
    frame = pd.read_csv(path, sep=r"\s+", skiprows=1, header=0, on_bad_lines="skip")
    if frame.shape[1] < 2:
        raise ValueError(f"Expected two numeric columns in {path}")
    wavelength = pd.to_numeric(frame.iloc[:, 0], errors="coerce").to_numpy(float)
    value = pd.to_numeric(frame.iloc[:, 1], errors="coerce").to_numpy(float)
    valid = np.isfinite(wavelength) & np.isfinite(value)
    wavelength, value = wavelength[valid], value[valid]
    order = np.argsort(wavelength)
    wavelength, value = wavelength[order], value[order]
    unique, inverse = np.unique(wavelength, return_inverse=True)
    if unique.size != wavelength.size:
        sums = np.bincount(inverse, weights=value)
        counts = np.bincount(inverse)
        wavelength, value = unique, sums / counts
    if wavelength.size < 3 or np.any(np.diff(wavelength) <= 0):
        raise ValueError(f"Invalid wavelength axis in {path}")
    return Spectrum(
        name=name or path.stem,
        path=path,
        wavelength_um=wavelength,
        value=value,
        sha256=sha256(path.read_bytes()).hexdigest(),
    )


def interpolate_supported(spectrum: Spectrum, wavelength_um: np.ndarray) -> np.ndarray:
    """Interpolate within measured support; return NaN outside it."""
    return np.interp(
        wavelength_um,
        spectrum.wavelength_um,
        spectrum.value,
        left=np.nan,
        right=np.nan,
    )


def build_components(
    component_spec: dict[str, dict[str, float]],
    input_dir: Path,
    grid_step_um: float,
    requested_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, list[str], list[Spectrum]]:
    """Build weighted component spectra on their common measured support."""
    loaded: dict[str, Spectrum] = {}
    for members in component_spec.values():
        for filename in members:
            if filename not in loaded:
                loaded[filename] = load_relab_spectrum(input_dir / filename)
    low = max(requested_range[0], *(s.support[0] for s in loaded.values()))
    high = min(requested_range[1], *(s.support[1] for s in loaded.values()))
    if high <= low:
        raise ValueError("Selected spectra have no common wavelength support")
    grid = np.arange(low, high + grid_step_um / 2, grid_step_um)
    columns: list[np.ndarray] = []
    names: list[str] = []
    for component, members in component_spec.items():
        weight_sum = float(sum(members.values()))
        if not np.isclose(weight_sum, 1.0, atol=1e-8):
            raise ValueError(f"Weights for {component} sum to {weight_sum}, not one")
        values = sum(
            weight * interpolate_supported(loaded[filename], grid)
            for filename, weight in members.items()
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Unsupported samples entered component {component}")
        columns.append(values)
        names.append(component)
    return grid, np.column_stack(columns), names, list(loaded.values())


def convolve_bands(
    grid_um: np.ndarray,
    spectra: np.ndarray,
    centers_um: np.ndarray,
    fwhm_um: np.ndarray,
) -> np.ndarray:
    """Integrate Gaussian spectral responses on a uniformly sampled grid."""
    result = np.empty((centers_um.size, spectra.shape[1]), dtype=float)
    for row, (center, width) in enumerate(zip(centers_um, fwhm_um, strict=True)):
        sigma = width / 2.354820045
        support = np.abs(grid_um - center) <= 4.0 * sigma
        if support.sum() < 3:
            raise ValueError(f"Insufficient samples around {center:.4f} um")
        weights = np.exp(-0.5 * ((grid_um[support] - center) / sigma) ** 2)
        denominator = np.trapz(weights, grid_um[support])
        result[row] = np.trapz(
            spectra[support] * weights[:, None], grid_um[support], axis=0
        ) / denominator
    return result
