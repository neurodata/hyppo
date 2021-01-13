from abc import ABC, abstractmethod

import numpy as np
from joblib import Parallel, delayed

from ..tools import compute_dist


class TimeSeriesTest(ABC):
    """
    Base class for time series in hyppo.

    Parameters
    ----------
    compute_distance : callable, optional (default: None)
        A function that computes the distance among the samples within each
        data matrix.
        Valid strings for ``metric`` are, as defined in
        ``sklearn.metrics.pairwise_distances``,

            - From scikit-learn: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’,
              ‘manhattan’] See the documentation for scipy.spatial.distance for details
              on these metrics.
            - From scipy.spatial.distance: [‘braycurtis’, ‘canberra’, ‘chebyshev’,
              ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’,
              ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
              ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’] See the
              documentation for scipy.spatial.distance for details on these metrics.

        Set to `None` or `precomputed` if `x` and `y` are already distance
        matrices. To call a custom function, either create the distance matrix
        before-hand or create a function of the form ``metric(x, **kwargs)``
        where `x` is the data matrix for which pairwise distances are
        calculated and kwargs are extra arguements to send to your custom function.

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

    def __init__(self, compute_distance=None, max_lag=0, **kwargs):
        # set statistic and p-value
        self.stat = None
        self.pvalue = None
        self.max_lag = max_lag
        self.kwargs = kwargs

        # set compute_distance kernel
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

    @abstractmethod
    def test(self, x, y, reps=1000, workers=1):
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
        self.distx, self.disty = compute_dist(
            x, y, metric=self.compute_distance, **self.kwargs
        )

        # calculate observed test statistic
        stat_list = self._statistic(x, y)
        stat = stat_list[0]

        # calculate null distribution
        null_dist = np.array(
            Parallel(n_jobs=workers)(
                [
                    delayed(_perm_stat)(self._statistic, self.distx, self.disty)
                    for rep in range(reps)
                ]
            )
        )
        pvalue = (1 + (null_dist >= stat).sum()) / (1 + reps)

        # correct for a p-value of 0. This is because, with bootstrapping
        # permutations, a p-value of 0 is incorrect
        if pvalue == 0:
            pvalue = 1 / reps
        self.pvalue = pvalue

        return stat, pvalue, stat_list


def _perm_stat(calc_stat, distx, disty):
    n = distx.shape[0]
    block_size = int(np.ceil(np.sqrt(n)))
    perm_index = np.r_[
        [np.arange(t, t + block_size) for t in np.random.choice(n, n // block_size + 1)]
    ].flatten()[:n]
    perm_index = np.mod(perm_index, n)
    permy = disty[np.ix_(perm_index, perm_index)]

    # calculate permuted statics, store in null distribution
    perm_stat = calc_stat(distx, permy)[0]

    return perm_stat
