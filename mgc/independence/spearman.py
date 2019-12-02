from scipy.stats import spearmanr

from .base import IndependenceTest
from ._utils import _CheckInputs


class Spearman(IndependenceTest):
    r"""
    Class for calculating the Spearman's :math:`\rho` test statistic and
    p-value.

    Spearman's :math:`\rho` coefficient is a nonparametric measure or rank
    correlation between two variables. It is equivalent to the Pearson's
    correlation with ranks.

    See Also
    --------
    Pearson : Pearson product-moment correlation test statistic and p-value.
    Kendall : Kendall's tau test statistic and p-value.

    Notes
    -----
    This class is a wrapper of `scipy.stats.spearmanr
    <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.
    spearmanr.html#scipy.stats.spearmanr>`_. The statistic can be derived
    as follows [#1Spea]_:

    Let :math:`x` and :math:`y` be :math:`(n, 1)` samples of random variables
    :math:`X` and :math:`Y`. Let :math:`rg_x` and :math:`rg_y` are the
    :math:`n` raw scores. Let :math:`\hat{\mathrm{cov}} (rg_x, rg_y)` is the
    sample covariance, and :math:`\hat{\sigma}_{rg_x}` and
    :math:`\hat{\sigma}_{rg_x}` are the sample variances of the rank
    variables. Then, the Spearman's :math:`\rho` coefficient is,

    .. math::

        \mathrm{Spearman}_n (x, y) =
            \frac{\hat{\mathrm{cov}} (rg_x, rg_y)}
            {\hat{\sigma}_{rg_x} \hat{\sigma}_{rg_y}}

    References
    ----------
    .. [#1Spea] Myers, J. L., Well, A. D., & Lorch Jr, R. F. (2013). *Research
                design and statistical analysis*. Routledge.
    """

    def __init__(self):
        IndependenceTest.__init__(self)

    def _statistic(self, x, y):
        r"""
        Helper function that calculates the Spearman's :math:`\rho` test
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
            The computed Spearman's rho statistic.
        """
        x.shape = (-1,)
        y.shape = (-1,)
        stat, _ = spearmanr(x, y)
        self.stat = stat

        return stat

    def test(self, x, y):
        r"""
        Calculates the Spearman's :math:`\rho` test statistic and p-value.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples and dimensions. That is, the shapes must be `(n, 1)` where
            `n` is the number of samples.

        Returns
        -------
        stat : float
            The computed Spearman's rho statistic.
        pvalue : float
            The computed Spearman's rho p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from mgc.independence import Spearman
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = Spearman().test(x, y)
        >>> '%.1f, %.2f' % (stat, pvalue)
        '1.0, 0.00'
        """
        check_input = _CheckInputs(x, y, dim=1)
        x, y = check_input()
        stat, pvalue = spearmanr(x, y)
        self.stat = stat
        self.pvalue = pvalue

        return stat, pvalue
