from __future__ import division

from builtins import zip, str, range, object
from past.utils import old_div

import autograd.numpy as np
import _utils, data, density, kernel, h0simulator
from .base import GofTest
from h0simulator import FSSDH0SimCovObs
import logging

import scipy
import scipy.stats as stats
from numpy.random import default_rng


class FSSD(GofTest):
    """
    Goodness-of-fit test using The Finite Set Stein Discrepancy statistic.
    and a set of paired test locations. The statistic is n*FSSD^2.
    The statistic can be negative because of the unbiased estimator.

    H0 : the sample follows p
    H1 : the sample does not follow p

    p is specified to the constructor in the form of an UnnormalizedDensity.
    """

    # NULLSIM_* are constants used to choose the way to simulate from the null
    # distribution to do the test.

    # Same as NULLSIM_COVQ; but assume that sample can be drawn from p.
    # Use the drawn sample to compute the covariance.
    NULLSIM_COVP = 1

    def __init__(
        self, p, k, V, null_sim=FSSDH0SimCovObs(n_simulate=3000, seed=101), alpha=0.01
    ):
        r"""
        Parameters
        ----------
        p : an instance of UnnormalizedDensity
        k : a DifferentiableKernel object
        V : J x dx numpy array of J locations to test the difference
        null_sim : an instance of H0Simulator for simulating from the null distribution.
        alpha : significance level
        """
        super(FSSD, self).__init__(p, alpha)
        self.k = k
        self.V = V
        self.null_sim = null_sim

    def test(self, dat, return_simulated_stats=False):
        r"""
        Parameters
        ----------
        dat : an instance of Data
        """
        alpha = self.alpha
        null_sim = self.null_sim
        n_simulate = null_sim.n_simulate
        X = dat.data()
        n = X.shape[0]
        J = self.V.shape[0]

        nfssd, fea_tensor = self.statistic(dat, return_feature_tensor=True)
        sim_results = null_sim.simulate(self, dat, fea_tensor)
        arr_nfssd = sim_results["sim_stats"]

        # approximate p-value with the permutations
        pvalue = np.mean(arr_nfssd > nfssd)

        results = {
            "alpha": self.alpha,
            "pvalue": pvalue,
            "test_stat": nfssd,
            "h0_rejected": pvalue < alpha,
            "n_simulate": n_simulate,
        }
        if return_simulated_stats:
            results["sim_stats"] = arr_nfssd
        return results

    def statistic(self, dat, return_feature_tensor=False):
        r"""
        Compute the test statistic. The statistic is n*FSSD^2.

        Parameters
        ----------
        dat : an instance of Data

        Returns
        -------
        stat : the test statistic, n*FSSD^2
        """
        X = dat.data()
        n = X.shape[0]

        # n x d x J
        Xi = self.feature_tensor(X)
        unscaled_mean = FSSD.ustat_h1_mean_variance(Xi, return_variance=False)
        stat = n * unscaled_mean

        if return_feature_tensor:
            return stat, Xi
        else:
            return stat

    def get_H1_mean_variance(self, dat):
        r"""
        Calculate the mean and variance under H1 of the test statistic (divided by
        n).

        Parameters
        ----------
        dat : an instance of Data

        Returns
        -------
        mean : the mean under the alternative hypothesis of the test statistic
        variance : the variance under the alternative hypothesis of the test statistic

        From: https://github.com/wittawatj/fsic-test
        """
        X = dat.data()
        Xi = self.feature_tensor(X)
        mean, variance = FSSD.ustat_h1_mean_variance(Xi, return_variance=True)
        return mean, variance

    def feature_tensor(self, X):
        r"""
        Compute the feature tensor which is n x d x J.
        The feature tensor can be used to compute the statistic, and the
        covariance matrix for simulating from the null distribution.

        Parameters
        ----------
        X : n x d data numpy array

        Returns
        -------
        an n x d x J numpy array

        From: https://github.com/wittawatj/fsic-test
        """
        k = self.k
        J = self.V.shape[0]
        n, d = X.shape
        # n x d matrix of gradients
        grad_logp = self.p.grad_log(X)

        K = k.eval(X, self.V)

        list_grads = np.array([np.reshape(k.gradX_y(X, v), (1, n, d)) for v in self.V])
        stack0 = np.concatenate(list_grads, axis=0)
        # a numpy array G of size n x d x J such that G[:, :, J]
        # is the derivative of k(X, V_j) with respect to X.
        dKdV = np.transpose(stack0, (1, 2, 0))

        # n x d x J tensor
        grad_logp_K = _utils.outer_rows(grad_logp, K)

        Xi = old_div((grad_logp_K + dKdV), np.sqrt(d * J))
        return Xi


def power_criterion(
    p, dat, k, test_locs, reg=1e-2, use_unbiased=True, use_2terms=False
):
    r"""
    Compute the mean and standard deviation of the statistic under H1.

    Parameters
    ----------
    use_2terms : True if the objective should include the first term in the power
        expression. This term carries the test threshold and is difficult to
        compute (depends on the optimized test locations). If True, then
        the objective will be -1/(n**0.5*sigma_H1) + n**0.5 FSSD^2/sigma_H1,
        which ignores the test threshold in the first term.

    Returns
    -------
    obj : mean/sd

    From: https://github.com/wittawatj/fsic-test
    """
    X = dat.data()
    n = X.shape[0]
    V = test_locs
    fssd = FSSD(p, k, V, null_sim=None)
    fea_tensor = fssd.feature_tensor(X)
    u_mean, u_variance = FSSD.ustat_h1_mean_variance(
        fea_tensor, return_variance=True, use_unbiased=use_unbiased
    )

    # mean/sd criterion
    sigma_h1 = np.sqrt(u_variance + reg)
    ratio = old_div(u_mean, sigma_h1)
    if use_2terms:
        obj = old_div(-1.0, (np.sqrt(n) * sigma_h1)) + np.sqrt(n) * ratio
    else:
        obj = ratio
    return obj


def ustat_h1_mean_variance(fea_tensor, return_variance=True, use_unbiased=True):
    r"""
    Compute the mean and variance of the asymptotic normal distribution
    under H1 of the test statistic.

    Parameters
    ----------
    fea_tensor : feature tensor obtained from feature_tensor()
    return_variance : If false, avoid computing and returning the variance.
    use_unbiased : If True, use the unbiased version of the mean. Can be
        negative.

    Returns
    -------
    stat : the mean
    variance: the variance

    From: https://github.com/wittawatj/fsic-test
    """
    Xi = fea_tensor
    n, d, J = Xi.shape

    assert n > 1, "Need n > 1 to compute the mean of the statistic."
    # n x d*J
    Tau = np.reshape(Xi, [n, d * J])
    if use_unbiased:
        t1 = np.sum(np.mean(Tau, 0) ** 2) * (old_div(n, float(n - 1)))
        t2 = old_div(np.sum(np.mean(Tau ** 2, 0)), float(n - 1))
        # stat is the mean
        stat = t1 - t2
    else:
        stat = np.sum(np.mean(Tau, 0) ** 2)

    if not return_variance:
        return stat

    # compute the variance
    # mu: d*J vector
    mu = np.mean(Tau, 0)
    variance = 4 * np.mean(np.dot(Tau, mu) ** 2) - 4 * np.sum(mu ** 2) ** 2
    return stat, variance


def list_simulate_spectral(cov, J, n_simulate=1000, seed=82):
    r"""
    Simulate the null distribution using the spectrums of the covariance
    matrix.  This is intended to be used to approximate the null
    distribution.

    Returns
    -------
    sim_fssds : a numpy array of simulated n*FSSD values
    eigs : eigenvalues of cov

    From: https://github.com/wittawatj/fsic-test
    """
    # eigen decompose
    eigs, _ = np.linalg.eig(cov)
    eigs = np.real(eigs)
    # sort in decreasing order
    eigs = -np.sort(-eigs)
    sim_fssds = FSSD.simulate_null_dist(eigs, J, n_simulate=n_simulate, seed=seed)
    return sim_fssds, eigs


def simulate_null_dist(eigs, J, n_simulate=2000, seed=7):
    r"""
    Simulate the null distribution using the spectrums of the covariance
    matrix of the U-statistic. The simulated statistic is n*FSSD^2 where
    FSSD is an unbiased estimator.

    Parameters
    ----------
    eigs : a numpy array of estimated eigenvalues of the covariance
        matrix. eigs is of length d*J, where d is the input dimension, and
    J : the number of test locations.

    Returns
    -------
    fssds : a numpy array of simulated statistics.

    From: https://github.com/wittawatj/fsic-test
    """
    d = old_div(len(eigs), J)
    assert d > 0
    # draw at most d x J x block_size values at a time
    block_size = max(20, int(old_div(1000.0, (d * J))))
    fssds = np.zeros(n_simulate)
    from_ind = 0
    rng = default_rng(seed)
    while from_ind < n_simulate:
        to_draw = min(block_size, n_simulate - from_ind)
        # draw chi^2 random variables.
        chi2 = rng.standard_normal(size=(d * J, to_draw)) ** 2
        # an array of length to_draw
        sim_fssds = eigs.dot(chi2 - 1.0)
        # store
        end_ind = from_ind + to_draw
        fssds[from_ind:end_ind] = sim_fssds
        from_ind = end_ind
    return fssds


def fssd_grid_search_kernel(p, dat, test_locs, list_kernel):
    r"""
    Linear search for the best kernel in the list that maximizes
    the test power criterion, fixing the test locations to V.

    Parameters
    ----------
    p : UnnormalizedDensity
    dat : a Data object
    list_kernel : list of kernel candidates

    Returns
    -------
    besti : best kernel index
    objs : array of test power criteria

    From: https://github.com/wittawatj/fsic-test
    """
    V = test_locs
    X = dat.data()
    n_cand = len(list_kernel)
    objs = np.zeros(n_cand)
    for i in range(n_cand):
        ki = list_kernel[i]
        objs[i] = FSSD.power_criterion(p, dat, ki, test_locs)
        logging.info("(%d), obj: %5.4g, k: %s" % (i, objs[i], str(ki)))

    # Widths that come early in the list
    # are preferred if test powers are equal.

    besti = objs.argmax()
    return besti, objs
