from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import cdist
from scipy._lib._util import MapWrapper

from .._utils import euclidean
from ._utils import k_sample_transform


class KSampleTest(ABC):
    """
    Base class for all tests in mgc.

    Parameters
    ----------
    compute_distance : callable, optional
        Function indicating distance metric (or alternatively the kernel) to
        use. Calculates the pairwise distance for each input, by default
        euclidean.

    Attributes
    ----------
    stat : float
        The computed independence test statistic.
    pvalue : float
        The computed independence test p-value.
    compute_distance : callable, optional
        Function indicating distance metric (or alternatively the kernel) to
        use. Calculates the pairwise distance for each input, by default
        euclidean.
    """

    def __init__(self, indep_test, compute_distance=None):
        # set statistic and p-value
        self.stat = None
        self.pvalue = None
        self.indep_test = indep_test

        # set compute_distance kernel
        if not compute_distance:
            compute_distance = euclidean
        self.compute_distance = compute_distance

        super().__init__()

    def _perm_stat(self, index):
        """
        Helper function that is used to calculate parallel permuted test
        statistics.

        Returns
        -------
        perm_stat : float
            Test statistic for each value in the null distribution.
        """
        u, v = k_sample_transform(self.inputs)
        permu = np.random.permutation(u)
        permv = np.random.permutation(v)

        # calculate permuted statics, store in null distribution
        perm_stat = self.indep_test.statistic(permu, permv)

        return perm_stat

    @abstractmethod
    def test(self, indep_test=None, reps=1000, workers=-1, *argv):
        """
        Calulates the independece test p-value.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices that have shapes depending on the particular
            independence tests (check desired test class for specifics).
        reps : int, optional
            The number of replications used in permutation, by default 1000.
        workers : int, optional
            Evaluates method using `multiprocessing.Pool <multiprocessing>`).
            Supply `-1` to use all cores available to the Process.

        Returns
        -------
        pvalue : float
            The pvalue obtained via permutation.
        null_dist : list
            The null distribution of the permuted test statistics.
        """
        self.inputs = list(range(*argv))
        self.indep_test = indep_test

        # calculate observed test statistic
        obs_stat = indep_test.statistic(*argv)

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
