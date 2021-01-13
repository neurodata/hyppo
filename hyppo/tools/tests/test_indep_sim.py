import numpy as np
import pytest
from numpy.testing import assert_equal

from .. import (
    circle,
    cubic,
    diamond,
    ellipse,
    exponential,
    fourth_root,
    joint_normal,
    linear,
    logarithmic,
    multimodal_independence,
    multiplicative_noise,
    quadratic,
    sin_four_pi,
    sin_sixteen_pi,
    spiral,
    square,
    step,
    two_parabolas,
    uncorrelated_bernoulli,
    w_shaped,
)


class TestIndepShape:
    @pytest.mark.parametrize("n", [100, 1000])
    @pytest.mark.parametrize("p", [1, 5])
    @pytest.mark.parametrize(
        "sim",
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
    def test_shapes(self, n, p, sim):
        np.random.seed(123456789)
        x, y = sim(n, p)
        nx, px = x.shape
        ny, py = y.shape

        if sim.__name__ in [
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
