import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from ..indep_sim import *
from ..ksample_sim import rot_2samp


class TestKSampleShape:
    @pytest.mark.parametrize("n", [10, 100, 1000])
    @pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("indep_sim", [
        linear, spiral
    ])
    @pytest.mark.parametrize("sim", [rot_2samp])
    def test_shapes(self, indep_sim, n, p, sim):
        np.random.seed(123456789)
        x, y = sim(indep_sim, n, p)
        nx, px = x.shape
        ny, py = y.shape

        assert_almost_equal(nx, ny, decimal=2)
        assert_almost_equal(px, 2, decimal=2)
        assert_almost_equal(py, 2, decimal=2)
