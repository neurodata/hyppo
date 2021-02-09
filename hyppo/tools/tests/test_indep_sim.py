import numpy as np
import pytest
from numpy.testing import assert_equal

from .. import SIMULATIONS, indep_sim


class TestIndepShape:
    @pytest.mark.parametrize("n", [100, 1000])
    @pytest.mark.parametrize("p", [1, 5])
    @pytest.mark.parametrize(
        "sim", SIMULATIONS.keys(),
    )
    def test_shapes(self, n, p, sim):
        np.random.seed(123456789)
        x, y = SIMULATIONS[sim](n, p)
        nx, px = x.shape
        ny, py = y.shape

        if sim in [
            "joint_normal",
            "logarithmic",
            "sin_four_pi",
            "sin_sixteen_pi",
            "two_parabolas",
            "square",
            "diamond",
            "circle",
            "ellipse",
            "multiplicative_noise",
            "multimodal_independence",
        ]:
            assert_equal(px, py)
        else:
            assert_equal(px, py * p)
        assert_equal(nx, ny)
