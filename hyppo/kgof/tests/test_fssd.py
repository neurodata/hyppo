import numpy as np
import numpy.testing as testing

from ..fssd import (
    FSSD,
    ustat_h1_mean_variance,
    power_criterion,
    fssd_grid_search_kernel,
    FSSDH0SimCovObs,
    FSSDH0SimCovDraw,
)
from .._utils import meddistance, fit_gaussian_draw, constrain
from ..kernel import KGauss
from ..density import IsotropicNormal

from numpy.random import default_rng

import scipy.stats as stats
import pytest


class TestFSSD:
    @pytest.mark.parametrize("n", [100])
    @pytest.mark.parametrize("alpha", [0.01])
    @pytest.mark.parametrize("d", [1])
    @pytest.mark.parametrize("J", [1])
    def test_basic(self, n, alpha, d, J):
        seed = 12
        # sample
        mean = np.zeros(d)
        variance = 1
        isonorm = IsotropicNormal(mean, variance)

        # only one dimension of the mean is shifted
        draw_mean = mean + 0
        draw_variance = variance + 1

        rng = default_rng(seed)
        X = rng.standard_normal(size=(n, d)) * np.sqrt(draw_variance) + draw_mean

        # Test
        sig2 = meddistance(X, subsample=1000) ** 2
        k = KGauss(sig2)
        list_k = [k]

        # random test locations
        V = fit_gaussian_draw(X, J, seed=seed + 1)
        null_sim = FSSDH0SimCovObs(n_simulate=200, seed=3)
        extra_sim = FSSDH0SimCovDraw()
        fssd = FSSD(isonorm, k, V, null_sim=null_sim, alpha=alpha)
        check_sim = extra_sim.simulate(gof=fssd)
        power_criterion(p=isonorm, X=X, k=k, test_locs=V)
        fssd_grid_search_kernel(p=isonorm, X=X, test_locs=V, list_kernel=list_k)
        fssd.get_H1_mean_variance(X=X)

        tresult = fssd.test(X, return_simulated_stats=True)

        # assertions
        testing.assert_almost_equal(tresult["pvalue"], 0, decimal=1)
        testing.assert_almost_equal(tresult["test_stat"], 1.6, decimal=1)

    @pytest.mark.parametrize("n", [200])
    @pytest.mark.parametrize("alpha", [0.01])
    @pytest.mark.parametrize("d", [4])
    @pytest.mark.parametrize("J", [1])
    def test_ustat_h1_mean_variance(self, n, alpha, d, J):
        seed = 20
        constrain(20, 19, 21)
        # sample
        mean = np.zeros(d)
        variance = 1
        isonorm = IsotropicNormal(mean, variance)

        draw_mean = mean + 2
        draw_variance = variance + 1
        rng = default_rng(seed)
        X = rng.standard_normal(size=(n, d)) * np.sqrt(draw_variance) + draw_mean

        # Test
        sig_square = meddistance(X, subsample=1000) ** 2
        k = KGauss(sig_square)

        # random test locations
        V = fit_gaussian_draw(X, J, seed=seed + 1)

        null_sim = FSSDH0SimCovObs(n_simulate=200, seed=3)
        fssd = FSSD(isonorm, k, V, null_sim=null_sim, alpha=alpha)
        fea_tensor = fssd.feature_tensor(X)

        u_mean, u_variance = ustat_h1_mean_variance(fea_tensor)

        random_sim = FSSDH0SimCovObs()
        random_sim.simulate(gof=fssd, X=X)

        # assertions
        testing.assert_almost_equal(u_variance, 1, decimal=1)
        # should reject H0
        testing.assert_almost_equal(u_mean, 0.8, decimal=1)
