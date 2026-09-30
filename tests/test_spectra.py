from pathlib import Path

import numpy as np

from lunar_band_design.spectra import interpolate_supported, load_relab_spectrum


def test_loader_does_not_extrapolate():
    path = Path(__file__).parents[1] / "relab/filtered_minerals/Troilite_LATB49.txt"
    spectrum = load_relab_spectrum(path)
    values = interpolate_supported(spectrum, np.array([0.1, 1.0, 30.0]))
    assert np.isnan(values[0])
    assert np.isfinite(values[1])
    assert np.isnan(values[2])
    assert spectrum.support[1] < 26.0
