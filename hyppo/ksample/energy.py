from ..independence.dcorr import _dcov
from ..tools import compute_dist
from ._utils import _CheckInputs
from .base import KSampleTest
from .ksamp import KSample


class Energy(KSampleTest):
    r"""
    Energy test statistic and p-value.

    Energy is a powerful multivariate 2-sample test. It leverages distance matrix
    capabilities (similar to tests like distance correlation or Dcorr). In fact, Energy
    statistic is equivalent to our 2-sample formulation nonparametric MANOVA via
    independence testing, i.e. :class:`hyppo.ksample.KSample`,
    and to
    :class:`hyppo.independence.Dcorr`,
    :class:`hyppo.ksample.DISCO`,
    :class:`hyppo.independence.Hsic`, and
    :class:`hyppo.ksample.MMD` `[1]`_ `[2]`_.

    Traditionally, the formulation for the 2-sample Energy statistic
    is as follows `[3]`_:

    Define
    :math:`\{ u_i \stackrel{iid}{\sim} F_U,\ i = 1, ..., n \}` and
    :math:`\{ v_j \stackrel{iid}{\sim} F_V,\ j = 1, ..., m \}` as two groups
    of samples deriving from different distributions with the same
    dimensionality. If :math:`d(\cdot, \cdot)` is a distance metric (i.e. euclidean)
    then,

    .. math::

        \mathrm{Energy}_{n, m}(\mathbf{u}, \mathbf{v}) = \frac{1}{n^2 m^2}
        \left( 2nm \sum_{i = 1}^n \sum_{j = 1}^m d(u_i, v_j) - m^2
        \sum_{i,j=1}^n d(u_i, u_j) - n^2 \sum_{i, j=1}^m d(v_i, v_j) \right)

    The implementation in the :class:`hyppo.ksample.KSample` class (using
    :class:`hyppo.independence.Dcorr` using 2 samples) is in
    fact equivalent to this implementation (for p-values) and statistics are
    equivalent up to a scaling factor `[1]`_.

    The p-value returned is calculated using a permutation test uses
    :meth:`hyppo.tools.perm_test`.
    The fast version of the test uses :meth:`hyppo.tools.chi2_approx`.

    .. _[1]: https://arxiv.org/abs/1910.08883
    .. _[2]: https://arxiv.org/abs/1806.05514
    .. _[3]: https://www.semanticscholar.org/paper/TESTING-FOR-EQUAL-DISTRIBUTIONS-IN-HIGH-DIMENSION-Sz%C3%A9kely-Rizzo/ad5e91905a85d6f671c04a67779fd1377e86d199

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
    bias : bool, default: False
        Whether or not to use the biased or unbiased test statistics.
    **kwargs
        Arbitrary keyword arguments for ``compute_distance``.
    """

    def __init__(self, compute_distance="euclidean", bias=False, **kwargs):
        KSampleTest.__init__(
            self, compute_distance=compute_distance, bias=bias, **kwargs
        )

    def statistic(self, x, y):
        r"""
        Calulates the Energy test statistic.

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
            The computed Energy statistic.
        """
        distx = x
        disty = y
        n = x.shape[0]
        m = y.shape[0]

        distx, disty = compute_dist(x, y, metric=self.compute_distance, **self.kwargs)

        # exact equivalence transformation Dcorr and Energy
        stat = (
            _dcov(distx, disty, self.bias) * (2 * (n ** 2) * (m ** 2)) / ((n + m) ** 4)
        )
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=1, auto=True):
        r"""
        Calculates the Energy test statistic and p-value.

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
            The computed Energy statistic.
        pvalue : float
            The computed Energy p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.ksample import Energy
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = Energy().test(x, y)
        >>> '%.3f, %.1f' % (stat, pvalue)
        '0.267, 1.0'
        """
        check_input = _CheckInputs(
            inputs=[x, y],
            indep_test="dcorr",
        )
        x, y = check_input()

        # observed statistic
        stat = self.statistic(x, y)

        # since stat transformation is invariant under permutation, 2-sample Dcorr
        # pvalue is identical to Energy
        _, pvalue = KSample(
            indep_test="Dcorr",
            compute_distkern=self.compute_distance,
            bias=self.bias,
            **self.kwargs
        ).test(x, y, reps=reps, workers=workers, auto=auto)

        return stat, pvalue
