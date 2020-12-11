import numpy as np
from math import ceil

from scipy._lib._util import check_random_state, MapWrapper

from hyppo.ksample._utils import k_sample_transform
from hyppo._utils import _PermGroups
from hyppo.sims.indep_sim import multilevel_gaussian


class _ParallelP(object):
    """
    Helper function to calculate parallel power.
    """

    def __init__(self, test, n, epsilon, d, rngs, blocks):
        self.test = test()
        self.n = n
        self.epsilon = epsilon
        self.d = d
        self.rngs = rngs
        self.blocks = blocks

    def __call__(self, index):
        # np.random.seed(self.rngs[index])
        x, y = multilevel_gaussian(n=self.n, epsilon=self.epsilon, d=self.d)

        u, v = k_sample_transform([x, y])
        obs_stat = self.test._statistic(u, v)
        permuter = _PermGroups(v, self.blocks)
        order = permuter()

        permv = v[order]

        # calculate permuted stats, store in null distribution
        perm_stat = self.test._statistic(u, permv)

        return obs_stat, perm_stat


def _perm_test(
    test,
    reps=1000,
    workers=-1,
    n=100,
    epsilon=0,
    d=2,
    blocks=None,
    random_state=None
):
    r"""
    Helper function that calculates the statistical.

    Parameters
    ----------
    test : callable()
        The independence test class requested.
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
    # set seeds
    random_state = check_random_state(random_state)
    rngs = [
        np.random.RandomState(random_state.randint(1 << 32, size=4, dtype=np.uint32))
        for _ in range(reps)
    ]

    # use all cores to create function that parallelizes over number of reps
    mapwrapper = MapWrapper(workers)
    parallelp = _ParallelP(
        test=test,
        n=n,
        epsilon=epsilon,
        d=d,
        rngs=rngs,
        blocks=blocks,
    )
    alt_dist, null_dist = map(list, zip(*list(mapwrapper(parallelp, range(reps)))))
    alt_dist = np.array(alt_dist)
    null_dist = np.array(null_dist)

    return alt_dist, null_dist


def power_2samp_multilevel(
    test,
    n=100,
    epsilon=0,
    d=2,
    blocks=None,
    reps=1000,
    workers=1,
    random_state=None,
    alpha=0.05,
):
    """
    [summary]

    Parameters
    ----------
    test : [type]
        [description]
    """

    alt_dist, null_dist = _perm_test(
        test=test,
        reps=reps,
        workers=workers,
        n=n,
        epsilon=epsilon,
        d=d,
        blocks=blocks,
    )
    cutoff = np.sort(null_dist)[ceil(reps * (1 - alpha))]
    empirical_power = (alt_dist >= cutoff).sum() / reps

    if empirical_power == 0:
        empirical_power = 1 / reps

    return empirical_power
