import numpy as np
from math import ceil

from scipy._lib._util import check_random_state, MapWrapper

from hyppo.ksample._utils import k_sample_transform
from hyppo.sims import gaussian_3samp


class _ParallelP3Samp(object):
    """
    Helper function to calculate parallel power.
    """

    def __init__(self, test, n, epsilon=1, weight=0, case=1, rngs=[]):
        self.test = test()

        self.n = n
        self.epsilon = epsilon
        self.weight = weight
        self.case = case
        self.rngs = rngs

    def __call__(self, index):
        if self.case not in [4, 5]:
            x, y, z = gaussian_3samp(self.n, epsilon=self.epsilon, case=self.case)
        else:
            x, y, z = gaussian_3samp(self.n, weight=self.weight, case=self.case)
        u, v = k_sample_transform([x, y, z])

        obs_stat = self.test._statistic(u, v)

        permv = self.rngs[index].permutation(v)

        # calculate permuted stats, store in null distribution
        perm_stat = self.test._statistic(u, permv)

        return obs_stat, perm_stat


def _perm_test_3samp(
    test, n=100, epsilon=1, weight=0, case=1, reps=1000, workers=1, random_state=None
):
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
    # set seeds
    random_state = check_random_state(random_state)
    rngs = [
        np.random.RandomState(random_state.randint(1 << 32, size=4, dtype=np.uint32))
        for _ in range(reps)
    ]

    # use all cores to create function that parallelizes over number of reps
    mapwrapper = MapWrapper(workers)
    parallelp = _ParallelP3Samp(test, n, epsilon, weight, case, rngs)
    alt_dist, null_dist = map(list, zip(*list(mapwrapper(parallelp, range(reps)))))
    alt_dist = np.array(alt_dist)
    null_dist = np.array(null_dist)

    return alt_dist, null_dist


def power_3samp_epsweight(
    test,
    n=100,
    epsilon=0.5,
    weight=0,
    case=1,
    alpha=0.05,
    reps=1000,
    workers=1,
    random_state=None,
):
    alt_dist, null_dist = _perm_test_3samp(
        test,
        n=n,
        epsilon=epsilon,
        weight=weight,
        case=case,
        reps=reps,
        workers=workers,
        random_state=random_state,
    )
    cutoff = np.sort(null_dist)[ceil(reps * (1 - alpha))]
    empirical_power = (alt_dist >= cutoff).sum() / reps

    if empirical_power == 0:
        empirical_power = 1 / reps

    return empirical_power
