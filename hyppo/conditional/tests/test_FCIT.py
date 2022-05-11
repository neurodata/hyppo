import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from hyppo.tools import rot_ksamp
from hyppo.conditional import FCIT


class TestFCIT:
    @pytest.mark.parametrize(
        "n, obs_stat, obs_pvalue",
        [
            (2000, 11.677197, 3.8168e-06),
            (1000, 7.733, 5.6549e-05),
        ],
    )
    def test_linear_oned(self, n, obs_stat, obs_pvalue):
        np.random.seed(123456789)
        x, y = rot_ksamp("linear", n, 1, k=2)
        np.random.seed(123456789)
        stat, pvalue = FCIT().test(x, y)

        assert_almost_equal(stat, obs_stat, decimal=-1)
        assert_almost_equal(pvalue, obs_pvalue, decimal=4)

    @pytest.mark.parametrize(
        "dim, n, obs_stat, obs_pvalue",
        [(1, 100000, -0.16024, 0.56139), (2, 100000, -4.59882, 0.99876)],
    )
    def test_null(self, dim, n, obs_stat, obs_pvalue):
        np.random.seed(12)
        z1 = np.random.multivariate_normal(
            mean=np.zeros(dim), cov=np.eye(dim), size=(n)
        )
        A1 = np.random.normal(loc=0, scale=1, size=dim * dim).reshape(dim, dim)
        B1 = np.random.normal(loc=0, scale=1, size=dim * dim).reshape(dim, dim)
        x1 = (
            A1 @ z1.T
            + np.random.multivariate_normal(
                mean=np.zeros(dim), cov=np.eye(dim), size=(n)
            ).T
        )
        y1 = (
            B1 @ z1.T
            + np.random.multivariate_normal(
                mean=np.zeros(dim), cov=np.eye(dim), size=(n)
            ).T
        )

        np.random.seed(122)
        stat, pvalue = FCIT().test(x1.T, y1.T, z1)

        assert_almost_equal(pvalue, obs_pvalue, decimal=4)
        assert_almost_equal(stat, obs_stat, decimal=4)

    @pytest.mark.parametrize(
        "dim, n, obs_stat, obs_pvalue",
        [
            (1, 100000, 89.271754, 2.91447597e-12),
            (2, 100000, 161.35165, 4.63412957e-14),
        ],
    )
    def test_alternative(self, dim, n, obs_stat, obs_pvalue):
        np.random.seed(12)
        z2 = np.random.multivariate_normal(
            mean=np.zeros(dim), cov=np.eye(dim), size=(n)
        )

        A2 = np.random.normal(loc=0, scale=1, size=dim * dim).reshape(dim, dim)
        B2 = np.random.normal(loc=0, scale=1, size=dim * dim).reshape(dim, dim)

        x2 = (
            A2 @ z2.T
            + np.random.multivariate_normal(
                mean=np.zeros(dim), cov=np.eye(dim), size=(n)
            ).T
        )
        y2 = (
            B2 @ x2
            + np.random.multivariate_normal(
                mean=np.zeros(dim), cov=np.eye(dim), size=(n)
            ).T
        )

        np.random.seed(122)
        stat, pvalue = FCIT().test(x2.T, y2.T, z2)

        assert_almost_equal(pvalue, obs_pvalue, decimal=12)
        assert_almost_equal(stat, obs_stat, decimal=4)
