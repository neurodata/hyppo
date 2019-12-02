from scipy.stats import pearsonr

from .base import IndependenceTest
from ._utils import _CheckInputs


class Pearson(IndependenceTest):
    r"""
    Class for calculating the Pearson test statistic and p-value.

    Pearson product-moment correlation coefficient is a measure of the linear
    correlation between two random variables [#1Pear]_. It has a value between
    +1 and -1 where 1 is the total positive linear correlation, 0 is not
    linear correlation, and -1 is total negative correlation.

    See Also
    --------
    RV : RV test statistic and p-value.
    CCA : CCA test statistic and p-value.
    Spearman : Spearman's rho test statistic and p-value.
    Kendall : Kendall's tau test statistic and p-value.

    Notes
    -----
    This class is a wrapper of `scipy.stats.pearsonr
    <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.
    pearsonr.html>`_. The statistic can be derived as follows [#1Pear]_:

    Let :math:`x` and :math:`y` be :math:`(n, 1)` samples of random variables
    :math:`X` and :math:`Y`. Let :math:`\hat{\mathrm{cov}} (x, y)` is the
    sample covariance, and :math:`\hat{\sigma}_x` and :math:`\hat{\sigma}_y`
    are the sample variances for :math:`x` and :math:`y`. Then, the Pearson's
    correlation coefficient is,

    .. math::

        \mathrm{Pearson}_n (x, y) =
            \frac{\hat{\mathrm{cov}} (x, y)}
            {\hat{\sigma}_x \hat{\sigma}_y}

    References
    ----------
    .. [#1Pear] Pearson, K. (1895). VII. Note on regression and inheritance in
                the case of two parents. *Proceedings of the Royal Society of
                London*, 58(347-352), 240-242.
    """

    def __init__(self):
        IndependenceTest.__init__(self)

    def _statistic(self, x, y):
        r"""
        Helper function that calculates the Pearson test statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples and dimensions. That is, the shapes must be `(n, 1)` where
            `n` is the number of samples.

        Returns
        -------
        stat : float
            The computed Pearson statistic.
        """
        x.shape = (-1,)
        y.shape = (-1,)
        stat, _ = pearsonr(x, y)
        self.stat = stat

        return stat

    def test(self, x, y):
        r"""
        Calculates the Pearson test statistic and p-value.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples and dimensions. That is, the shapes must be `(n, 1)` where
            `n` is the number of samples.

        Returns
        -------
        stat : float
            The computed Pearson statistic.
        pvalue : float
            The computed Pearson p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from mgc.independence import Pearson
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = Pearson().test(x, y)
        >>> '%.1f, %.2f' % (stat, pvalue)
        '1.0, 0.00'
        """
        check_input = _CheckInputs(x, y, dim=1)
        x, y = check_input()
        stat, pvalue = pearsonr(x, y)
        self.stat = stat
        self.pvalue = pvalue

        return stat, pvalue
