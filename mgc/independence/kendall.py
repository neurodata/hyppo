from scipy.stats import kendalltau

from .base import IndependenceTest
from ._utils import _CheckInputs


class Kendall(IndependenceTest):
    r"""
    Class for calculating the Kendall's :math:`\tau` test statistic and
    p-value.

    Kendall's :math:`\tau` coefficient is a statistic to meassure ordinal
    associations between two quantities. The Kendall's :math:`\tau`
    correlation between high when variables similar rank relative to other
    observations [#1Kend]_. Both this and the closely related Spearman's
    :math:`\rho` coefficient are special cases of a general correlation
    coefficient.

    See Also
    --------
    Pearson : Pearson product-moment correlation test statistic and p-value.
    Spearman : Spearman's rho test statistic and p-value.

    Notes
    -----
    This class is a wrapper of `scipy.stats.kendalltau
    <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats
    .kendalltau.html#scipy.stats.kendalltau>`_. The statistic can be derived
    as follows [#1Kend]_:

    Let :math:`x` and :math:`y` be :math:`(n, 1)` samples of random variables
    :math:`X` and :math:`Y`. Define :math:`(x_i, y_i)` and :math:`(x_j, y_j)`
    as concordant if the ranks agree: :math:`x_i > x_j` and :math:`y_i > y_j`
    or `x_i > x_j` and :math:`y_i < y_j`. They are discordant if the ranks
    disagree: :math:`x_i > x_j` and :math:`y_i < y_j` or :math:`x_i < x_j` and
    :math:`y_i > y_j`. If :math:`x_i > x_j` and :math:`y_i < y_j`, the pair is
    said to be tied. Let :math:`n_c` and :math:`n_d` be the number of
    concordant and discordant pairs respectively and :math:`n_0 = n(n-1) / 2`.
    In the case of no ties, the test statistic is defined as

    .. math::

        \mathrm{Kendall}_n (x, y) = \frac{n_c - n_d}{n_0}

    Further, define :math:`n_1 = \sum_i \frac{t_i (t_i - 1)}{2}`,
    :math:`n_2 = \sum_j \frac{u_j (u_j - 1)}{2}`, :math:`t_i` be the number of
    tied values in the :math:`i`th group and :math:`u_j` be the number of tied
    values in the :math:`j`th group. Then, the statistic is [#2Kend]_,

    .. math::

        \mathrm{Kendall}_n (x, y) = \frac{n_c - n_d}
                                         {\sqrt{(n_0 - n_1) (n_0 - n_2)}}

    References
    ----------
    .. [#1Kend] Kendall, M. G. (1938). A new measure of rank correlation.
                *Biometrika*, 30(1/2), 81-93.
    .. [#2Kend] Agresti, A. (2010). *Analysis of ordinal categorical data*
                (Vol. 656). John Wiley & Sons.
    """

    def __init__(self):
        IndependenceTest.__init__(self)

    def _statistic(self, x, y):
        r"""
        Helper function that calculates the Kendall's :math:`\tau` test
        statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples and dimensions. That is, the shapes must be `(n, 1)` where
            `n` is the number of samples.

        Returns
        -------
        stat : float
            The computed Kendall's tau statistic.
        """
        x.shape = (-1,)
        y.shape = (-1,)
        stat, _ = kendalltau(x, y)
        self.stat = stat

        return stat

    def test(self, x, y):
        r"""
        Calculates the Kendall's :math:`\tau` test statistic and p-value.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples and dimensions. That is, the shapes must be `(n, 1)` where
            `n` is the number of samples.

        Returns
        -------
        stat : float
            The computed Kendall's tau statistic.
        pvalue : float
            The computed Kendall's tau p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from mgc.independence import Kendall
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = Kendall().test(x, y)
        >>> '%.1f, %.2f' % (stat, pvalue)
        '1.0, 0.00'
        """
        check_input = _CheckInputs(x, y, dim=1)
        x, y = check_input()
        stat, pvalue = kendalltau(x, y)
        self.stat = stat
        self.pvalue = pvalue

        return stat, pvalue
