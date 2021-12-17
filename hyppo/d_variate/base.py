from abc import ABC, abstractmethod
from typing import NamedTuple

from ..tools import multi_perm_test


class DVariateTestOutput(NamedTuple):
    stat: float
    pvalue: float


class DVariateTest(ABC):
    r"""
    A base class for a :math:`d`-variate independence test.

    Parameters
    ----------
    compute_kernel : str, callable, or None, default: "gaussian"
        A function that computes the kernel similarity among the samples within each
        data matrix.
        Valid strings for ``compute_kernel`` are, as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`,

            [``"additive_chi2"``, ``"chi2"``, ``"linear"``, ``"poly"``,
            ``"polynomial"``, ``"rbf"``,
            ``"laplacian"``, ``"sigmoid"``, ``"cosine"``]

        Note ``"rbf"`` and ``"gaussian"`` are the same metric.
        Set to ``None`` or ``"precomputed"`` if ``args`` are already similarity
        matrices. To call a custom function, either create the similarity matrix
        before-hand or create a function of the form :func:`metric(x, **kwargs)`
        where ``x`` is the data matrix for which pairwise kernel similarity matrices are
        calculated and kwargs are extra arguments to send to your custom function.
    **kwargs
        Arbitrary keyword arguments for ``multi_compute_kern``.
    """

    def __init__(self, compute_kernel=None, **kwargs):
        # set statistic and p-value
        self.stat = None
        self.pvalue = None
        self.compute_kernel = compute_kernel
        self.kwargs = kwargs

        super().__init__()

    @abstractmethod
    def statistic(self, *args):
        r"""
        Calculates the :math:`d`-variate independence test statistic.

        Parameters
        ----------
        *args: ndarray of float
            Variable length input data matrices. All inputs must have the same
            number of samples. That is, the shapes must be ``(n, p)``, ``(n, q)``,
            etc., where `n` is the number of samples and `p` and `q` are the
            number of dimensions.
        """

    @abstractmethod
    def test(self, *args, reps=1000, workers=1):
        r"""
        Calculates the d_variate independence test statistic and p-value.

        Parameters
        ----------
        *args : ndarray of float
            Variable length input data matrices. All inputs must have the same
            number of samples. That is, the shapes must be ``(n, p)``, ``(n, q)``,
            etc., where `n` is the number of samples and `p` and `q` are the
            number of dimensions.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed :math:`d`-variate independence test statistic.
        pvalue : float
            The computed :math:`d`-variate independence p-value.
        """
        self.args = args

        stat, pvalue, null_dist = multi_perm_test(
            self.statistic,
            *args,
            reps=reps,
            workers=workers,
        )
        self.stat = stat
        self.pvalue = pvalue
        self.null_dist = null_dist

        return DVariateTestOutput(stat, pvalue)
