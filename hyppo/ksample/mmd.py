from ._utils import _CheckInputs
from .base import KSampleTest
from .ksamp import KSample


class MMD(KSampleTest):
    r"""
    Maximum Mean Discrepency (MMD) test statistic and p-value.

    MMD is a powerful multivariate 2-sample test. It leverages kernel similarity
    matrices
    capabilities (similar to tests like distance correlation or Dcorr). In fact, MMD
    statistic is equivalent to our 2-sample formulation nonparametric MANOVA via
    independence testing, i.e. :class:`hyppo.ksample.KSample`,
    and to
    :class:`hyppo.independence.Dcorr`,
    :class:`hyppo.ksample.DISCO`,
    :class:`hyppo.independence.Hsic`, and
    :class:`hyppo.ksample.Energy` `[1]`_ `[2]`_.

    Traditionally, the formulation for the 2-sample MMD statistic
    is as follows `[3]`_:

    Define
    :math:`\{ u_i \stackrel{iid}{\sim} F_U,\ i = 1, ..., n \}` and
    :math:`\{ v_j \stackrel{iid}{\sim} F_V,\ j = 1, ..., m \}` as two groups
    of samples deriving from different distributions with the same
    dimensionality. If :math:`k(\cdot, \cdot)` is a kernel metric (i.e. gaussian)
    then,

    .. math::

        \mathrm{MMD}_{n, m}(\mathbf{u}, \mathbf{v}) =
        \frac{1}{m(m - 1)} \sum_{i = 1}^m \sum_{j \neq i}^m k(u_i, u_j)
        + \frac{1}{n(n - 1)} \sum_{i = 1}^n \sum_{j \neq i}^n k(v_i, v_j)
        - \frac{2}{mn} \sum_{i = 1}^n \sum_{j \neq i}^n k(v_i, v_j)

    The implementation in the :class:`hyppo.ksample.KSample` class (using
    :class:`hyppo.independence.Hsic` using 2 samples) is in
    fact equivalent to this implementation (for p-values) and statistics are
    equivalent up to a scaling factor `[1]`_.

    The p-value returned is calculated using a permutation test uses
    :meth:`hyppo.tools.perm_test`.
    The fast version of the test uses :meth:`hyppo.tools.chi2_approx`.

    .. _[1]: https://arxiv.org/abs/1910.08883
    .. _[2]: https://arxiv.org/abs/1806.05514
    .. _[3]: https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

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
        calculated and kwargs are extra arguements to send to your custom function.
    bias : bool, default: False
        Whether or not to use the biased or unbiased test statistics.
    **kwargs
        Arbitrary keyword arguments for ``compute_kernel``.
    """

    def __init__(self, compute_kernel="gaussian", bias=False, **kwargs):
        self.compute_kernel = compute_kernel
        self.bias = bias
        KSampleTest.__init__(self, compute_distance=None, bias=bias, **kwargs)

    def statistic(self, x, y):
        r"""
        Calulates the MMD test statistic.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            dimensions. That is, the shapes must be ``(n, p)`` and ``(m, p)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions.

        Returns
        -------
        stat : float
            The computed MMD statistic.
        """
        n = x.shape[0]
        m = y.shape[0]

        # exact equivalence transformation Hsic and MMD
        stat = (
            KSample(
                indep_test="Hsic",
                compute_distkern=self.compute_kernel,
                bias=self.bias,
                **self.kwargs
            ).statistic(
                x,
                y,
            )
            * (2 * (n ** 2) * (m ** 2))
            / ((n + m) ** 4)
        )
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=1, auto=True):
        r"""
        Calculates the MMD test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            dimensions. That is, the shapes must be ``(n, p)`` and ``(m, p)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.
        auto : bool, default: True
            Automatically uses fast approximation when `n` and size of array
            is greater than 20. If ``True``, and sample size is greater than 20, then
            :class:`hyppo.tools.chi2_approx` will be run. Parameters ``reps`` and
            ``workers`` are
            irrelevant in this case. Otherwise, :class:`hyppo.tools.perm_test` will be
            run.

        Returns
        -------
        stat : float
            The computed MMD statistic.
        pvalue : float
            The computed MMD p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.ksample import MMD
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = MMD().test(x, y)
        >>> '%.3f, %.1f' % (stat, pvalue)
        '-0.015, 1.0'
        """
        check_input = _CheckInputs(
            inputs=[x, y],
            indep_test="hsic",
        )
        x, y = check_input()

        # observed statistic
        stat = self.statistic(x, y)

        # since stat transformation is invariant under permutation, 2-sample Hsic
        # pvalue is identical to MMD
        _, pvalue = KSample(
            indep_test="Hsic",
            compute_distkern=self.compute_kernel,
            bias=self.bias,
            **self.kwargs
        ).test(x, y, reps=reps, workers=workers, auto=auto)

        return stat, pvalue
