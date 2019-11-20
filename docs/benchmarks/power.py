import sys, os
import multiprocessing as mp
from itertools import product

import numpy as np
from math import ceil

from mgc.independence import *
from mgc.sims.indep_sim import *

from scipy._lib._util import MapWrapper


class _ParallelP(object):
    """
    Helper function to calculate parallel power.
    """
    def __init__(self, test, sim, n, p, noise):
        self.test = test()
        self.sim = sim

        self.n = n
        self.p = p
        self.noise = noise

    def __call__(self, index):
        if (self.sim.__name__ == "multiplicative_noise" or
            self.sim.__name__ == "multimodal_independence"):
            x, y = self.sim(self.n, self.p)
        else:
            x, y = self.sim(self.n, self.p, noise=self.noise)

        obs_stat = self.test._statistic(x, y)

        permx = np.random.permutation(x)
        permy = np.random.permutation(y)

        # calculate permuted stats, store in null distribution
        perm_stat = self.test._statistic(permx, permy)

        obs_stat = np.abs(obs_stat)
        perm_stat = np.abs(perm_stat)

        return obs_stat, perm_stat


def _perm_test(test, sim, n=100, p=1, noise=False, reps=1000, workers=-1):
    r"""
    Helper function that calculates the statistical.

    Parameters
    ----------
    test : callable()
        The independence test class requested.
    sim : callable()
        The simulation used to generate the input data.
    reps : int, optional (default: 1000)
        The number of replications used to estimate the null distribution
        when using the permutation test used to calculate the p-value.
    workers : int, optional (default: -1)
        The number of cores to parallelize the p-value computation over.
        Supply -1 to use all cores available to the Process.

    Returns
    -------
    null_dist : list
        The approximated null distribution.
    """

    # use all cores to create function that parallelizes over number of reps
    mapwrapper = MapWrapper(workers)
    parallelp = _ParallelP(test=test, sim=sim, n=n, p=p, noise=noise)
    alt_dist, null_dist = map(list, zip(*list(mapwrapper(parallelp, range(reps)))))
    alt_dist = np.array(alt_dist)
    null_dist = np.array(null_dist)

    return alt_dist, null_dist


def power_sample(test, sim, n=100, p=1, noise=True, alpha=0.05, reps=1000, workers=1):
    """
    [summary]

    Parameters
    ----------
    test : [type]
        [description]
    sim : [type]
        [description]
    n : int, optional
        [description], by default 100
    p : int, optional
        [description], by default 1
    noise : int, optional
        [description], by default 0
    reps : int, optional
        [description], by default 1000
    alpha : float, optional
        [description], by default 0.05
    """

    alt_dist, null_dist = _perm_test(test, sim, n=n, p=p, noise=noise,
                                     reps=reps, workers=workers)
    cutoff = np.sort(null_dist)[ceil(reps * (1-alpha))]
    empirical_power = (alt_dist >= cutoff).sum() / reps

    if empirical_power == 0:
        empirical_power = 1 / reps

    return empirical_power


def indep_power_sampsize():
    sys.path.append(os.path.realpath('..'))
    os.system("taskset -p 0xff %d" % os.getpid())

    SAMP_SIZES = range(5, 100, 5)
    POWER_REPS = 3

    simulations = [linear, exponential, cubic, joint_normal, step, quadratic,
                   w_shaped, spiral, uncorrelated_bernoulli, logarithmic,
                   fourth_root, sin_four_pi, sin_sixteen_pi, square,
                   two_parabolas, circle, ellipse, diamond,
                   multiplicative_noise, multimodal_independence]
    tests = [CCA, Dcorr, HHG, Hsic, Kendall, Pearson, RV, Spearman]

    def estimate_power(sim, test):
        print('{} {} started'.format(sim.__name__, test.__name__))
        est_power = np.array([np.mean([power_sample(test, sim, n=i)
                              for _ in range(POWER_REPS)])
                              for i in SAMP_SIZES])
        np.savetxt('../benchmarks/vs_samplesize/{}_{}.csv'.format(
            sim.__name__, test.__name__), est_power, delimiter=',')
        print('{} {} finished'.format(sim.__name__, test.__name__))

    with mp.Pool() as p:
        p.starmap(estimate_power, product(simulations, tests))
