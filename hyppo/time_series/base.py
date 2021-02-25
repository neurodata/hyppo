from abc import ABC, abstractmethod

import numpy as np
from joblib import Parallel, delayed

from ..tools import compute_dist


class TimeSeriesTest(ABC):
    """
    A base class for a time-series test.

    Parameters
    ----------
    compute_distance : str, callable, or None, default: "euclidean"
        A function that computes the distance among the samples within each
        data matrix.
        Valid strings for ``compute_distance`` are, as defined in
        :func:`sklearn.metrics.pairwise_distances`,

            - From scikit-learn: [``"euclidean"``, ``"cityblock"``, ``"cosine"``,
              ``"l1"``, ``"l2"``, ``"manhattan"``] See the documentation for
              :mod:`scipy.spatial.distance` for details
              on these metrics.
            - From scipy.spatial.distance: [``"braycurtis"``, ``"canberra"``,
              ``"chebyshev"``, ``"correlation"``, ``"dice"``, ``"hamming"``,
              ``"jaccard"``, ``"kulsinski"``, ``"mahalanobis"``, ``"minkowski"``,
              ``"rogerstanimoto"``, ``"russellrao"``, ``"seuclidean"``,
              ``"sokalmichener"``, ``"sokalsneath"``, ``"sqeuclidean"``,
              ``"yule"``] See the documentation for :mod:`scipy.spatial.distance` for
              details on these metrics.

        Set to ``None`` or ``"precomputed"`` if ``x`` and ``y`` are already distance
        matrices. To call a custom function, either create the distance matrix
        before-hand or create a function of the form ``metric(x, **kwargs)``
        where ``x`` is the data matrix for which pairwise distances are
        calculated and ``**kwargs`` are extra arguements to send to your custom
        function.
    max_lag : float, default: 0
        The maximium lag to consider when computing the test statistics and p-values.
    **kwargs
        Arbitrary keyword arguments for ``compute_distance``.
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
    def statistic(self, x, y):
        """
        Calulates the time-series test statistic.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions. Alternatively, ``x`` and ``y`` can be distance matrices,
            where the shapes must both be ``(n, n)``.
        """

    @abstractmethod
    def test(self, x, y, reps=1000, workers=1):
        """
        Calulates the time-series test test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, p)`` and ``(n, q)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions. Alternatively, ``x`` and ``y`` can be distance matrices,
            where the shapes must both be ``(n, n)``.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.

        Returns
        -------
        stat : float
            The discriminability test statistic.
        pvalue : float
            The discriminability p-value.
        null_dist : list
            The null distribution of the permuted test statistics.
        """
        distx, disty = compute_dist(x, y, metric=self.compute_distance, **self.kwargs)

        # calculate observed test statistic
        stat_list = self.statistic(x, y)
        stat = stat_list[0]

        # calculate null distribution
        null_dist = np.array(
            Parallel(n_jobs=workers)(
                [
                    delayed(_perm_stat)(self.statistic, distx, disty)
                    for rep in range(reps)
                ]
            )
        )
        pvalue = (1 + (null_dist >= stat).sum()) / (1 + reps)
        self.pvalue = pvalue
        self.null_dist = null_dist

        return stat, pvalue, stat_list


def _perm_stat(calc_stat, distx, disty):
    """Permutes the test statistics."""
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
