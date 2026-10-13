import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from ...tools import linear, power
from .. import Dcorr


class TestDcorrStat:
    @pytest.mark.parametrize("n", [100, 200])
    @pytest.mark.parametrize("obs_stat", [1.0])
    @pytest.mark.parametrize("obs_pvalue", [1 / 1000])
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = linear(n, 1)
        stat1, pvalue1 = Dcorr().test(x, y)
        stat2 = Dcorr().statistic(x, y)

        assert_almost_equal(stat1, obs_stat, decimal=2)
        assert_almost_equal(stat2, obs_stat, decimal=2)
        assert_almost_equal(pvalue1, obs_pvalue, decimal=2)

    @pytest.mark.parametrize("n", [100, 200])
    def test_rep(self, n):
        x, y = linear(n, 1)
        stat1, pvalue1 = Dcorr().test(x, y, random_state=2)
        stat2, pvalue2 = Dcorr().test(x, y, random_state=2)

        assert stat1 == stat2
        assert pvalue1 == pvalue2

    def test_dcorr_sqrt_bug(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 9, 4, 4])
        stat = Dcorr(bias=True).test(x, y, reps=0)[0]

        assert_almost_equal(stat, 0.762676242417, decimal=2)

    def test_fast_large_n_no_overflow(self):
        # n**4 overflows int64 for n > ~55,000; the fast statistic of independent
        # data must stay near 0 instead of blowing up (issue #353)
        rng = np.random.default_rng(0)
        n = 70000
        x = rng.standard_normal((n, 1))
        y = rng.standard_normal((n, 1))
        test = Dcorr()
        test.is_fast = True
        stat = test.statistic(x, y)

        assert abs(stat) < 1e-2

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_dtypes(self, dtype):
        np.random.seed(123456789)
        x, y = linear(100, 1)
        x_cast = (x * 100).astype(dtype)
        y_cast = (y * 100).astype(dtype)
        stat = Dcorr().statistic(x_cast, y_cast)
        assert_almost_equal(stat, 1.0, decimal=2)


class TestDcorrTypeIError:
    def test_oned(self):
        np.random.seed(123456789)
        est_power = power(
            "Dcorr",
            sim_type="indep",
            sim="multimodal_independence",
            n=100,
            p=1,
            alpha=0.05,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)

    def test_oned_fast(self):
        np.random.seed(123456789)
        est_power = power(
            "Dcorr",
            sim_type="indep",
            sim="multimodal_independence",
            n=100,
            p=1,
            alpha=0.05,
            auto=True,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)

    def test_threed_fast(self):
        np.random.seed(123456789)
        est_power = power(
            "Dcorr",
            sim_type="indep",
            sim="multimodal_independence",
            n=100,
            p=3,
            alpha=0.05,
            auto=True,
        )

        assert_almost_equal(est_power, 0.05, decimal=2)
