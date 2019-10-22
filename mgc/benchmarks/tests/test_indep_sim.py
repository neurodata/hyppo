import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from ..indep_sim import linear, spiral


class TestIndepShape:
    @pytest.mark.parametrize("n", [10, 100, 1000])
    @pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("sim", [
        linear, spiral
    ])
    def test_shapes(self, n, p, sim):
        np.random.seed(123456789)
        x, y = sim(n, p)
        nx, px = x.shape
        ny, py = y.shape

        assert_almost_equal(nx, ny, decimal=2)
        assert_almost_equal(px, py*p, decimal=2)
