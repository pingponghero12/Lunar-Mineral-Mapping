import numpy as np

from lunar_band_design.unmixing import project_simplex


def test_simplex_projection():
    projected = project_simplex(np.array([-0.2, 0.3, 1.4]))
    assert np.all(projected >= 0)
    assert np.isclose(projected.sum(), 1.0)
