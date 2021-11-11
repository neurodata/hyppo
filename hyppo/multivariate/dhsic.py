from .base import MultivariateTest, MultivariateTestOutput
from ..tools import multi_compute_kern
from ._utils import _CheckInputs

import numpy as np


class Dhsic(MultivariateTest):
    r"""
    d-variate Hilbert Schmidt Independence Criterion (Dhsic) test statistic
    and p-value.

    The d-variable Hilbert Schmidt independence criterion (Dhsic) is a
    non-parametric kernel-based independence test between an arbitrary number
    of variables. The Dhsic statistic is 0 if the variables are jointly
    independent and positive if the variables are dependent.
    :footcite:p:`grettonKernelJointIndependence2016`.
    The default choice is the Gaussian kernel, which uses the median distance
    as the bandwidth, which is a characteristic kernel that guarantees that
    Dhsic is a consistent test
    :footcite:p:`grettonKernelStatisticalTest2007`
    :footcite:p:`grettonConsistentNonparametricTests2010`
    :footcite:p:`grettonKernelJointIndependence2016`.

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
        Set to ``None`` or ``"precomputed"`` if ``x`` and ``y`` are already similarity
        matrices. To call a custom function, either create the similarity matrix
        before-hand or create a function of the form :func:`metric(x, **kwargs)`
        where ``x`` is the data matrix for which pairwise kernel similarity matrices are
        calculated and kwargs are extra arguments to send to your custom function.
    bias : bool, default: False
        Whether or not to use the biased or unbiased test statistics.
    **kwargs
        Arbitrary keyword arguments for ``multi_compute_kern``.

    Notes
    -----
    The statistic can be derived as follows
    :footcite:p:`grettonKernelJointIndependence2016`:

    Dhsic builds on the two-variable Hilbert Schmidt Independence Criterion (Hsic),
    implemented in :class:`hyppo.independence.Hsic`, but allows for an arbitrary
    number of variables. For a given kernel, the joint distribution and the product
    of the marginals is mapped to the reproducing kernel Hilbert space and the
    squared distance between the embeddings is calculated.

    References
    ----------
    .. footbibliography::
    """
    def __init__(self, compute_kernel="gaussian", bias=True, **kwargs):
        self.compute_kernel = compute_kernel
        self.bias = bias

        MultivariateTest.__init__(self, compute_distance=None, **kwargs)

    def statistic(self, *data_matrices):
        """
        Helper function that calculates the Dhsic test statistic.

        Parameters
        ----------
        *data_matrices: Tuple[np.ndarray]
            Input data matrices. All elements of the tuple must have the same
            number of samples. That is, the shapes must be ``(n, p)``, ``(n, q)``,
            etc., where `n` is the number of samples and `p` and `q` are the
            number of dimensions. Alternatively, the elements can be distance
            matrices, where the shapes must both be ``(n, n)``.

        Returns
        -------
        stat : float
            The computed Dhsic statistic.
        """
        kerns = multi_compute_kern(*data_matrices, metric=self.compute_kernel, **self.kwargs)

        n = kerns[0].shape[0]
        term1 = np.ones((n, n))
        term2 = 1
        term3 = (2 / n) * np.ones((n, ))
        for j in range(len(kerns)):
            term1 = np.multiply(term1, kerns[j])
            term2 = (1 / n ** 2) * term2 * np.sum(kerns[j])
            term3 = (1 / n) * np.multiply(term3, np.sum(kerns[j], axis=1))

        stat = (1 / n ** 2) * np.sum(term1) + term2 - np.sum(term3)
        self.stat = stat

        return stat

    def test(self, *data_matrices, reps=1000, workers=1):
        """
        Calculates the Dhsic test statistic and p-value.

        Parameters
        ----------
        *data_matrices: Tuple[np.ndarray]
            Input data matrices. All elements of the tuple must have the same
            number of samples. That is, the shapes must be ``(n, p)``, ``(n, q)``,
            etc., where `n` is the number of samples and `p` and `q` are the
            number of dimensions. Alternatively, the elements can be distance
            matrices, where the shapes must both be ``(n, n)``.
        reps : int, default=1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default=1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed dHsic statistic.
        pvalue : float
            The computed dHsic p-value.
        """
        check_input = _CheckInputs(
            *data_matrices,
            reps=reps,
        )
        data_matrices = check_input()

        stat, pvalue = super(Dhsic, self).test(*data_matrices, reps=reps, workers=workers)

        return MultivariateTestOutput(stat, pvalue)

