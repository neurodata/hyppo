import numpy as np

from ..independence.dcorr import _dcov
from ..tools import compute_dist
from ._utils import _CheckInputs, k_sample_transform
from .base import KSampleTest
from .ksamp import KSample


class DISCO(KSampleTest):
    r"""
    Distance Components (DISCO) test statistic and p-value.

    DISCO is a powerful multivariate `k`-sample test. It leverages distance matrix
    capabilities (similar to tests like distance correlation or Dcorr). In fact, DISCO
    statistic is equivalent to our 2-sample formulation nonparametric MANOVA via
    independence testing, i.e. :class:`hyppo.ksample.KSample`,
    and to
    :class:`hyppo.independence.Dcorr`,
    :class:`hyppo.ksample.Energy`,
    :class:`hyppo.independence.Hsic`, and
    :class:`hyppo.ksample.MMD` `[1]`_ `[2]`_.

    Traditionally, the formulation for the DISCO statistic
    is as follows `[3]`_:

    Define
    :math:`\{ u^i_1 \stackrel{iid}{\sim} F_{U_1},\ i = 1, ..., n_1 \}` up to
    :math:`\{ u^j_k \stackrel{iid}{\sim} F_{V_1},\ j = 1, ..., n_k \}` as `k` groups
    of samples deriving from different distributions with the same
    dimensionality. If :math:`d(\cdot, \cdot)` is a distance metric (i.e. euclidean),
    :math:`N = \sum_{i = 1}^k n_k`,
    and :math:`\mathrm{Energy}` is the Energy test statistic from
    :class:`hyppo.ksample.Energy`
    then,

    .. math::

        \mathrm{DISCO}_N(\mathbf{u}_1, \ldots, \mathbf{u}_k) =
        \sum_{1 \leq k < l \leq K} \frac{n_k n_l}{2N}
        \mathrm{Energy}_{n_k + n_l} (\mathbf{u}_k, \mathbf{u}_l)

    The implementation in the :class:`hyppo.ksample.KSample` class (using
    :class:`hyppo.independence.Dcorr`) is in
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

    def statistic(self, *args):
        r"""
        Calulates the DISCO test statistic.

        Parameters
        ----------
        *args : ndarray
            Variable length input data matrices. All inputs must have the same
            number of samples and dimensions. That is, the shapes must be `(n, p)`
            where `n` are the number of samples and `p` is
            the number of dimensions.

        Returns
        -------
        stat : float
            The computed DISCO statistic.
        """
        inputs = list(args)
        N = [i.shape[0] for i in inputs]

        if len(set(N)) > 1:
            raise ValueError(
                "Shape mismatch, inputs must have same sample size, "
                "currently {}".format(len(set(N)))
            )
        u, v = k_sample_transform(inputs)

        distu, distv = compute_dist(u, v, metric=self.compute_distance, **self.kwargs)

        # exact equivalence transformation Dcov and DISCO
        stat = _dcov(distu, distv, self.bias) * np.sum(N) * len(N) / 2
        self.stat = stat

        return stat

    def test(self, *args, reps=1000, workers=1, auto=True):
        r"""
        Calculates the DISCO test statistic and p-value.

        Parameters
        ----------
        *args : ndarray
            Variable length input data matrices. All inputs must have the same
            number of samples and dimensions. That is, the shapes must be `(n, p)`
            where `n` is the number of samples and `p` is
            the number of dimensions.
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
            The computed DISCO statistic.
        pvalue : float
            The computed DISCO p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.ksample import DISCO
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = DISCO().test(x, y)
        >>> '%.3f, %.1f' % (stat, pvalue)
        '-1.566, 1.0'
        """
        inputs = list(args)
        check_input = _CheckInputs(
            inputs=inputs,
            indep_test="dcorr",
        )
        inputs = check_input()
        N = [i.shape[0] for i in inputs]

        if len(set(N)) > 1:
            raise ValueError(
                "Shape mismatch, inputs must have same sample size, "
                "currently {}".format(len(set(N)))
            )

        # observed statistic
        stat = self.statistic(*inputs)

        # since stat transformation is invariant under permutation, k-sample Dcorr
        # pvalue is identical to DISCO
        _, pvalue = KSample(
            indep_test="Dcorr",
            compute_distkern=self.compute_distance,
            bias=self.bias,
            **self.kwargs
        ).test(*inputs, reps=reps, workers=workers, auto=auto)

        return stat, pvalue
