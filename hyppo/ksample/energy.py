import numpy as np
from numba import njit

from ..independence.dcorr import _center_distmat
from ..tools import compute_dist
from ._utils import _CheckInputs
from .base import KSampleTest
from .ksamp import KSample


class Energy(KSampleTest):
    r"""
    Class for calculating the Energy test statistic and p-value.

    Energy is a powerful multivariate 2-sample test. It leverages distance matrix
    capabilities (similar to tests like distance correlation or Dcorr). In fact, Energy
    statistic is equivalent to our 2-sample formulation nonparametric MANOVa via
    independence testing, i.e. ``hyppo.ksample.Ksample``,
    and to Dcorr, Hilbert Schmidt Independence Criterion (Hsic), and Maximum Mean
    Discrepancy [#1Ener]_ [#2Ener]_  (see "See Also" section for links).

    Parameters
    ----------
    bias : bool (default: False)
        Whether or not to use the biased or unbiased test statistics.

    Notes
    -----
    Traditionally, the formulation for the 2-sample Energy statistic
    is as follows [#3Ener]_:

    Define
    :math:`\{ u_i \stackrel{iid}{\sim} F_U,\ i = 1, ..., n \}` and
    :math:`\{ v_j \stackrel{iid}{\sim} F_V,\ j = 1, ..., m \}` as two groups
    of samples deriving from different distributions with the same
    dimensionality. If :math:`d(\cdot, \cdot)` is a distance metric (i.e. euclidean)
    then,

    .. math::

        Energy_{n, m}(\mathbf{u}, \mathbf{v}) = \frac{1}{n^2 m^2}
        \left( 2nm \sum_{i = 1}^n \sum_{j = 1}^m d(u_i, v_j) - m^2
        \sum_{i,j=1}^n d(u_i, u_j) - n^2 \sum_{i, j=1}^m d(v_i, v_j) \right)

    The implementation in the ``hyppo.ksample.KSample`` class (using Dcorr) is in
    fact equivalent to this implementation (for p-values) and statistics are
    equivalent up to a scaling factor [#1Ener]_.

    The p-value returned is calculated using a permutation test using a
    `permutation test <https://hyppo.neurodata.io/reference/tools.html#permutation-test>`_.
    The fast version of the test (for :math:`k`-sample Dcorr and Hsic) uses a
    `chi squared approximation <https://hyppo.neurodata.io/reference/tools.html#chi-squared-approximation>`_.

    References
    ----------
    .. [#1Ener] Panda, S., Shen, C., Perry, R., Zorn, J., Lutz, A., Priebe, C. E., &
                Vogelstein, J. T. (2019). Nonparametric MANOVA via Independence
                Testing. arXiv e-prints, arXiv-1910.
    .. [#2Ener] Shen, C., & Vogelstein, J. T. (2018). The exact equivalence of distance
                and kernel methods for hypothesis testing. arXiv preprint
                arXiv:1806.05514.
    .. [#3Ener] Székely, G. J., & Rizzo, M. L. (2004). Testing for equal distributions
                in high dimension. InterStat, 5(16.10), 1249-1272.
    """

    def __init__(self, compute_distance="euclidean", bias=False, **kwargs):
        # set is_distance to true if compute_distance is None
        self.is_distance = False
        if not compute_distance:
            self.is_distance = True
        KSampleTest.__init__(
            self, compute_distance=compute_distance, bias=bias, **kwargs
        )

    def _statistic(self, x, y):
        r"""
        Calulates the Energy test statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples. That is, the shapes must be `(n, p)` and `(n, q)` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions. Alternatively, `x` and `y` can be distance matrices,
            where the shapes must both be `(n, n)`.
        """
        distx = x
        disty = y
        n = x.shape[0]
        m = y.shape[0]

        if not self.is_distance:
            distx, disty = compute_dist(
                x, y, metric=self.compute_distance, **self.kwargs
            )

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
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples. That is, the shapes must be `(n, p)` and `(n, q)` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions. Alternatively, `x` and `y` can be distance matrices,
            where the shapes must both be `(n, n)`.
        reps : int, optional (default: 1000)
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, optional (default: 1)
            The number of cores to parallelize the p-value computation over.
            Supply -1 to use all cores available to the Process.
        auto : bool (default: True)
            Automatically uses fast approximation when sample size and size of array
            is greater than 20. If True, and sample size is greater than 20, a fast
            chi2 approximation will be run. Parameters ``reps`` and ``workers`` are
            irrelevant in this case.

        Returns
        -------
        stat : float
            The computed *k*-Sample statistic.
        pvalue : float
            The computed *k*-Sample p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.ksample import Energy
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = Energy().test(x, y)
        >>> '%.3f, %.1f' % (stat, pvalue)
        '0.000, 1.0'
        """
        check_input = _CheckInputs(
            inputs=[x, y],
            indep_test="dcorr",
        )
        x, y = check_input()

        # observed statistic
        stat = Energy()._statistic(x, y)

        # since stat transformation is invariant under permutation, 2-sample Dcorr
        # pvalue is identical to Energy
        _, pvalue = KSample("Dcorr").test(x, y, reps=reps, workers=workers, auto=auto)

        return stat, pvalue


@njit
def _dcov(distx, disty, bias):  # pragma: no cover
    """Calculate the Dcorr test statistic"""
    # center distance matrices
    cent_distx = _center_distmat(distx, bias)
    cent_disty = _center_distmat(disty, bias)

    N = distx.shape[0]

    if bias:
        stat = 1 / (N ** 2) * np.trace(np.multiply(cent_distx, cent_disty))
    else:
        stat = 1 / (N * (N - 3)) * np.trace(np.multiply(cent_distx, cent_disty))

    return stat
