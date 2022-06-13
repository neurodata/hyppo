import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_raises

from ...tools import power, rot_ksamp
from .. import MANOVA, Hotelling


class TestManova:
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue",
        [(1000, 0.005062841807278008, 1.0), (100, 8.24e-5, 0.9762956529114515)],
    )
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2, noise=False)
        results = MANOVA().test(x, y)

        assert_almost_equal(results["Pillai's trace"]["statistic"], obs_stat, decimal=1)
        assert_almost_equal(results["Pillai's trace"]["p-value"], obs_pvalue, decimal=1)

    def test_twosample_manova(self):
        rng = np.random.default_rng(123456789)
        n = 1000
        p = 20
        x1 = rng.standard_normal((n, p))
        x2 = rng.standard_normal((n, p))
        _, pvalue_hotelling = Hotelling().test(x1, x2)
        results_manova = MANOVA().test(x1, x2)
        assert_almost_equal(
            pvalue_hotelling, results_manova["Pillai's trace"]["p-value"], decimal=4
        )


class TestManovaErrorWarn:
    """Tests errors and warnings derived from MGC."""

    def test_no_indeptest(self):
        # raises error if not indep test
        x = np.arange(100).reshape(5, 20)
        y = np.arange(50, 150).reshape(5, 20)
        assert_raises(ValueError, MANOVA().test, x, y)


class TestManovaTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "MANOVA",
            sim_type="gauss",
            sim="multimodal_independence",
            case=1,
            n=100,
            alpha=0.05,
        )

        assert est_power <= 0.05
