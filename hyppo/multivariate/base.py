from abc import ABC, abstractmethod
from typing import NamedTuple

from ..tools import multi_perm_test


class MultivariateTestOutput(NamedTuple):
    stat: float
    pvalue: float


class MultivariateTest(ABC):
    r"""
    A base class for a multivariate independence test.

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
    **kwargs
        Arbitrary keyword arguments for ``multi_compute_kern``.
    """

    def __init__(self, compute_distance=None, **kwargs):
        # set statistic and p-value
        self.stat = None
        self.pvalue = None
        self.compute_distance = compute_distance
        self.kwargs = kwargs

        super().__init__()

    @abstractmethod
    def statistic(self, *data_matrices):
        r"""
        Calculates the multivariate independence test statistic.

        Parameters
        ----------
        *data_matrices: Tuple[np.ndarray]
            Input data matrices. All elements of the tuple must have the same
            number of samples. That is, the shapes must be ``(n, p)``, ``(n, q)``,
            etc., where `n` is the number of samples and `p` and `q` are the
            number of dimensions. Alternatively, the elements can be distance
            matrices, where the shapes must both be ``(n, n)``.
        """

    @abstractmethod
    def test(self, *data_matrices, reps=1000, workers=1):
        r"""
        Calculates the multivariate independence test statistic and p-value.

        Parameters
        ----------
        *data_matrices : Tuple[np.ndarray]
            Input data matrices. All elements of the tuple must have the same
            number of samples. That is, the shapes must be ``(n, p)``, ``(n, q)``,
            etc., where `n` is the number of samples and `p` and `q` are the
            number of dimensions. Alternatively, the elements can be distance
            matrices, where the shapes must both be ``(n, n)``.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed multivariate independence test statistic.
        pvalue : float
            The computed multivariate independence p-value.
        """
        self.data_matrices = data_matrices

        stat, pvalue, null_dist = multi_perm_test(
            self.statistic,
            *data_matrices,
            reps=reps,
            workers=workers,
        )
        self.stat = stat
        self.pvalue = pvalue
        self.null_dist = null_dist

        return MultivariateTestOutput(stat, pvalue)
