import numpy as np
import numpy.testing as testing

from ..fssd import (
    FSSD,
    FSSDH0SimCovObs,
    FSSDH0SimCovDraw,
    ustat_h1_mean_variance,
    power_criterion,
    fssd_grid_search_kernel,
)
from .._utils import meddistance, fit_gaussian_draw, constrain
from ..kernel import KGauss
from ..data import Data
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
        dat = Data(X)
        dat.dim()
        dat.sample_size()
        dat.n()
        dat.split_tr_te()
        dat.subsample(n=4)
        dat.clone()

        # Test
        sig2 = meddistance(X, subsample=1000) ** 2
        k = KGauss(sig2)
        list_k = [k]

        # random test locations
        V = fit_gaussian_draw(X, J, seed=seed + 1)
        null_sim = FSSDH0SimCovObs(n_simulate=200, seed=3)
        extra_sim = FSSDH0SimCovDraw()
        fssd = FSSD(isonorm, k, V, null_sim=null_sim, alpha=alpha)
        check_sim = extra_sim.simulate(dat=dat, gof=fssd)
        power_criterion(p=isonorm, dat=dat, k=k, test_locs=V)
        fssd_grid_search_kernel(p=isonorm, dat=dat, test_locs=V, list_kernel=list_k)
        fssd.get_H1_mean_variance(dat=dat)

        tresult = fssd.test(dat, return_simulated_stats=True)
        dat.__add__(dat)

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
        dat = Data(X)

        # Test
        sig2 = meddistance(X, subsample=1000) ** 2
        k = KGauss(sig2)

        # random test locations
        V = fit_gaussian_draw(X, J, seed=seed + 1)

        null_sim = FSSDH0SimCovObs(n_simulate=200, seed=3)
        fssd = FSSD(isonorm, k, V, null_sim=null_sim, alpha=alpha)
        fea_tensor = fssd.feature_tensor(X)

        u_mean, u_variance = ustat_h1_mean_variance(fea_tensor)

        # assertions
        testing.assert_almost_equal(u_variance, 1, decimal=1)
        # should reject H0
        testing.assert_almost_equal(u_mean, 0.8, decimal=1)
