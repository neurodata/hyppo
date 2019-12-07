from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import cdist
from scipy._lib._util import check_random_state, MapWrapper

from .._utils import euclidean


class TimeSeriesTest(ABC):
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

    def __init__(self, compute_distance=None, max_lag=0):
        # set statistic and p-value
        self.stat = None
        self.pvalue = None
        self.max_lag = max_lag

        # set compute_distance kernel
        if not compute_distance:
            compute_distance = euclidean
        self.compute_distance = compute_distance

        super().__init__()

    @abstractmethod
    def _statistic(self, x, y):
        """
        Calulates the independence test statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices that have shapes depending on the particular
            independence tests (check desired test class for specifics).
        """

    def _perm_stat(self, index):                                                # pragma: no cover
        """
        Helper function that is used to calculate parallel permuted test
        statistics.

        Returns
        -------
        perm_stat : float
            Test statistic for each value in the null distribution.
        """
        n = self.distx.shape[0]
        perm_index = np.r_[[np.arange(t, t+self.block_size) for t in
                            self.rngs[index].choice(n,
                            n//self.block_size + 1)]].flatten()[:n]
        perm_index = np.mod(perm_index, n)
        permx = self.distx[np.ix_(perm_index, perm_index)]
        permy = self.disty[np.ix_(perm_index, perm_index)]

        # calculate permuted statics, store in null distribution
        perm_stat = self._statistic(permx, permy)

        return perm_stat

    @abstractmethod
    def test(self, x, y, reps=1000, workers=1, random_state=None):
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
        random_state : int or np.random.RandomState instance, optional
            If already a RandomState instance, use it.
            If seed is an int, return a new RandomState instance seeded with seed.
            If None, use np.random.RandomState. Default is None.

        Returns
        -------
        pvalue : float
            The pvalue obtained via permutation.
        null_dist : list
            The null distribution of the permuted test statistics.
        """
        self.distx = self.compute_distance(x)
        self.disty = self.compute_distance(y)

        # calculate observed test statistic
        obs_stat = self._statistic(x, y)

        # generate seeds for each rep (change to new parallel random number
        # capabilities in numpy >= 1.17+)
        random_state = check_random_state(random_state)
        self.rngs = [np.random.RandomState(random_state.randint(1 << 32,
                     size=4, dtype=np.uint32)) for _ in range(reps)]
        n = x.shape[0]
        self.block_size = int(np.ceil(np.sqrt(n)))

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
