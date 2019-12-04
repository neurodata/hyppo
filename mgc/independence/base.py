from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.distance import cdist, squareform
from scipy._lib._util import check_random_state, MapWrapper

from .._utils import euclidean


class IndependenceTest(ABC):
    r"""
    A base class for an independence test.

    Parameters
    ----------
    compute_distance : callable(), optional (default: euclidean)
        A function that computes the distance or similarity among the samples
        within each data matrix. Set to `None` if `x` and `y` are already
        distance matrices. To call a custom function, either create the
        distance matrix before-hand or create a function of the form
        ``compute_distance(x)`` where `x` is the data matrix for which
        pairwise distances are calculated.
    """

    def __init__(self, compute_distance=euclidean):
        # set statistic and p-value
        self.stat = None
        self.pvalue = None
        self.compute_distance = compute_distance

        super().__init__()

    @abstractmethod
    def _statistic(self, x, y):
        r"""
        Calulates the independence test statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices.
        """

    def _perm_stat(self, index):                                                # pragma: no cover
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
        permx = self.rngs[index].permutation(self.x)
        permy = self.rngs[index].permutation(self.y)

        # calculate permuted statics, store in null distribution
        perm_stat = self._statistic(permx, permy)

        return perm_stat

    @abstractmethod
    def test(self, x, y, reps=1000, workers=1, random_state=None):
        r"""
        Calulates the independence test p-value.

        Parameters
        ----------
        x, y : ndarray
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
            The computed independence test statistic.
        pvalue : float
            The pvalue obtained via permutation.
        """

        self.x = x
        self.y = y

        # calculate observed test statistic
        stat = self._statistic(x, y)

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
        pvalue = (null_dist >= stat).sum() / reps

        # correct for a p-value of 0. This is because, with bootstrapping
        # permutations, a p-value of 0 is incorrect
        if pvalue == 0:
            pvalue = 1 / reps
        self.pvalue = pvalue

        return stat, pvalue
