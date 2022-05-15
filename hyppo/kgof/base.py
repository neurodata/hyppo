from __future__ import division

from abc import ABC, abstractmethod


class GofTest(ABC):
    r"""
    A base class for a discriminability test.

    Parameters
    ----------
    compute_distance : str, callable, or None, default: "euclidean" or "gaussian"
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
        Alternatively, this function computes the kernel similarity among the
        samples within each data matrix.
        Valid strings for ``compute_kernel`` are, as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`,
            [``"additive_chi2"``, ``"chi2"``, ``"linear"``, ``"poly"``,
            ``"polynomial"``, ``"rbf"``,
            ``"laplacian"``, ``"sigmoid"``, ``"cosine"``]
        Note ``"rbf"`` and ``"gaussian"`` are the same metric.
    """

    def __init__(self, p, alpha):
        """
        p : an UnnormalizedDensity object
        alpha : float, significance level of the test
        """
        self.p = p
        self.alpha = alpha

    @abstractmethod
    def test(self, X):
        """
        Perform the goodness-of-fit test and return values
        computed in a dictionary.

        Parameters
        ----------
        dat : an instance of Data (observed data)

        Returns
        -------
        {
            alpha: 0.01,
            pvalue: 0.0002,
            test_stat: 2.3,
            h0_rejected: True,
            time_secs: ...
        }
        """
        raise NotImplementedError()

    @abstractmethod
    def statistic(self, X):
        r"""
        Calculates the goodness-of-fit test statistic.

        Parameters
        ----------
        dat : an instance of Data (observed data)
            Input data matrices.
        """
        raise NotImplementedError()
