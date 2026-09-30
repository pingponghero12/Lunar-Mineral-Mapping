import numpy as np

from lunar_band_design.sensor import fwhm_um, valid_candidate_centers


def test_resolution_regimes_and_support_margin():
    widths = fwhm_um(np.array([1.0, 10.0, 20.0, 40.0]))
    assert np.allclose(widths, [0.01, 0.1941176471, 0.6, 1.2])
    centers, fwhm = valid_candidate_centers(0.4, 25.0, 0.02)
    assert np.all(centers - 1.70 * fwhm >= 0.4)
    assert np.all(centers + 1.70 * fwhm <= 25.0)
