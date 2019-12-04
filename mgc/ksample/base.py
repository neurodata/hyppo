from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import cdist
from scipy._lib._util import check_random_state, MapWrapper

from .._utils import euclidean
from ._utils import k_sample_transform
from ..independence import Dcorr, HHG, Hsic


class KSampleTest(ABC):
    """
    A base class for a k-sample test.

    Parameters
    ----------
    indep_test : {CCA, Dcorr, HHG, RV, Hsic}
        The class corresponding to the desired independence test from
        ``mgc.independence``.
    compute_distance : callable(), optional (default: euclidean)
        A function that computes the distance or similarity among the samples
        within each data matrix. Set to `None` if `x` and `y` are already
        distance matrices. To call a custom function, either create the
        distance matrix before-hand or create a function of the form
        ``compute_distance(x)`` where `x` is the data matrix for which
        pairwise distances are calculated.
    """

    def __init__(self, indep_test, compute_distance=euclidean):
        # set statistic and p-value
        self.stat = None
        self.pvalue = None
        self.compute_distance = compute_distance

        dist_tests = [Dcorr, HHG, Hsic]
        if indep_test in dist_tests:
            self.indep_test = indep_test(compute_distance=compute_distance)
        else:
            self.indep_test = indep_test()

        super().__init__()

    def _perm_stat(self, index):                                            # pragma: no cover
        r"""
        Helper function that is used to calculate parallel permuted test
        statistics.

        Parameters
        ----------
        index : int
            Iterator used for parallel statistic calculation

        Returns
        -------
        perm_stat : float
            Test statistic for each value in the null distribution.
        """

        permu = self.rngs[index].permutation(self.u)
        permv = self.rngs[index].permutation(self.v)

        # calculate permuted statics, store in null distribution
        perm_stat = self.indep_test._statistic(permu, permv)

        return perm_stat

    @abstractmethod
    def test(self, inputs, reps=1000, workers=1, random_state=None):
        r"""
        Calulates the k-sample test p-value.

        Parameters
        ----------
        inputs : list of ndarray
            Input data matrices.
        reps : int, optional
            The number of replications used in permutation, by default 1000.
        workers : int, optional (default: 1)
            Evaluates method using `multiprocessing.Pool <multiprocessing>`).
            Supply `-1` to use all cores available to the Process.
        random_state : int or np.random.RandomState instance, optional
            If already a RandomState instance, use it.
            If seed is an int, return a new RandomState instance seeded with seed.
            If None, use np.random.RandomState. Default is None.

        Returns
        -------
        stat : float
            The computed k-sample test statistic.
        pvalue : float
            The pvalue obtained via permutation.
        """

        # calculate observed test statistic
        u, v = k_sample_transform(inputs)
        self.u = u
        self.v = v
        obs_stat = self.indep_test._statistic(u, v)

        # generate seeds for each rep (change to new parallel random number
        # capabilities in numpy >= 1.17+)
        random_state = check_random_state(random_state)
        self.rngs = [np.random.RandomState(random_state.randint(1 << 32,
                     size=4, dtype=np.uint32)) for _ in range(reps)]

        # use all cores to create function that parallelizes over number of reps
        mapwrapper = MapWrapper(workers)
        null_dist = np.array(list(mapwrapper(self._perm_stat, range(reps))))
        self.null_dist = null_dist

        # calculate p-value and significant permutation map through list
        pvalue = (null_dist >= obs_stat).sum() / reps

        # correct for a p-value of 0. This is because, with bootstrapping
        # permutations, a p-value of 0 is incorrect
        if pvalue == 0:
            pvalue = 1 / reps
        self.pvalue = pvalue

        return obs_stat, pvalue
