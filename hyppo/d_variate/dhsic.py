from .base import DVariateTest, DVariateTestOutput
from ..tools import multi_compute_kern
from ._utils import _CheckInputs

import numpy as np


class dHsic(DVariateTest):
    r"""
    :math:`d`-variate Hilbert Schmidt Independence Criterion (dHsic) test
    statistic and p-value.

    dHsic is a non-parametric kernel-based independence test between an
    arbitrary number of variables. The dHsic statistic is 0 if the variables
    are jointly independent and positive if the variables are dependent
    :footcite:p:`grettonKernelJointIndependence2016`.
    The default choice is the Gaussian kernel, which uses the median distance
    as the bandwidth, which is a characteristic kernel that guarantees that
    dHsic is a consistent test
    :footcite:p:`grettonKernelJointIndependence2016`
    :footcite:p:`grettonKernelStatisticalTest2007`
    :footcite:p:`grettonConsistentNonparametricTests2010`.

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
    bias : bool, default: False
        Whether or not to use the biased or unbiased test statistics.
    **kwargs
        Arbitrary keyword arguments for ``multi_compute_kern``.

    Notes
    -----
    The statistic can be derived as follows
    :footcite:p:`grettonKernelJointIndependence2016`:

    dHsic builds on the two-variable Hilbert Schmidt Independence Criterion (Hsic),
    implemented in :class:`hyppo.independence.Hsic`, but allows for an arbitrary
    number of variables. For a given kernel, the joint distribution and the product
    of the marginals is mapped to the reproducing kernel Hilbert space and the
    squared distance between the embeddings is calculated. The dHsic statistic can
    be calculated by,

    .. math::

        \mathrm{dHsic} (\mathbb{P}^{(X^1, ..., X^d)}) = \Big\Vert \Pi(\mathbb{P}^{X^1}
        \otimes \cdot\cdot\cdot \otimes \mathbb{P}^{X^d}) - \Pi(\mathbb{P}^
        {(X^1, ..., X^d)}) \Big\Vert^2_{\mathscr{H}}

    Similar to Hsic, dHsic uses a gaussian median kernel by default, and the p-value
    is calculated using a permutation test using :meth:`hyppo.tools.multi_perm_test`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, compute_kernel="gaussian", bias=True, **kwargs):
        self.compute_kernel = compute_kernel
        self.bias = bias

        DVariateTest.__init__(self, compute_kernel=self.compute_kernel, **kwargs)

    def statistic(self, *args):
        """
        Helper function that calculates the dHsic test statistic.

        Parameters
        ----------
        *args: ndarray of float
            Variable length input data matrices. All inputs must have the same
            number of samples. That is, the shapes must be ``(n, p)``, ``(n, q)``,
            etc., where `n` is the number of samples and `p` and `q` are the
            number of dimensions.

        Returns
        -------
        stat : float
            The computed dHsic statistic.
        """
        kerns = multi_compute_kern(*args, metric=self.compute_kernel, **self.kwargs)

        n = kerns[0].shape[0]
        term1 = np.ones((n, n))
        term2 = 1
        term3 = (2 / n) * np.ones((n,))
        for kern in kerns:
            term1 = np.multiply(term1, kern)
            term2 = (1 / n**2) * term2 * np.sum(kern)
            term3 = (1 / n) * np.multiply(term3, np.sum(kern, axis=1))

        stat = (1 / n**2) * np.sum(term1) + term2 - np.sum(term3)
        self.stat = stat

        return stat

    def test(self, *args, reps=1000, workers=1):
        """
        Calculates the dHsic test statistic and p-value.

        Parameters
        ----------
        *args: ndarray of float
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
            The computed dHsic statistic.
        pvalue : float
            The computed dHsic p-value.
        """
        check_input = _CheckInputs(
            *args,
            reps=reps,
        )
        args = check_input()

        stat, pvalue = super(dHsic, self).test(*args, reps=reps, workers=workers)
        self.stat = stat
        self.pvalue = pvalue

        return DVariateTestOutput(stat, pvalue)
