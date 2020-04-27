import pytest
import numpy as np
from numpy.testing import assert_equal

from .. import (
    linear,
    spiral,
    exponential,
    cubic,
    joint_normal,
    step,
    quadratic,
    w_shaped,
    uncorrelated_bernoulli,
    logarithmic,
    fourth_root,
    sin_four_pi,
    sin_sixteen_pi,
    two_parabolas,
    circle,
    ellipse,
    diamond,
    multiplicative_noise,
    square,
    multimodal_independence,
    rot_2samp,
    trans_2samp,
)


class TestKSampleShape:
    @pytest.mark.parametrize("n", [100, 1000])
    @pytest.mark.parametrize("p", [1, 5])
    @pytest.mark.parametrize(
        "indep_sim",
        [
            linear,
            spiral,
            exponential,
            cubic,
            joint_normal,
            step,
            quadratic,
            w_shaped,
            uncorrelated_bernoulli,
            logarithmic,
            fourth_root,
            sin_four_pi,
            sin_sixteen_pi,
            two_parabolas,
            circle,
            ellipse,
            diamond,
            multiplicative_noise,
            square,
            multimodal_independence,
        ],
    )
    @pytest.mark.parametrize("sim", [rot_2samp, trans_2samp])
    def test_shapes(self, indep_sim, n, p, sim):
        np.random.seed(123456789)
        x, y = sim(indep_sim, n, p)
        nx, px = x.shape
        ny, py = y.shape

        sim_name = indep_sim.__name__
        if sim_name in [
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
            assert_equal(px, p * 2)
            assert_equal(py, p * 2)
        else:
            assert_equal(px, p + 1)
            assert_equal(py, p + 1)
        assert_equal(nx, ny)
