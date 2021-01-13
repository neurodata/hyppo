import numpy as np

from ._utils import _CheckInputs
from .base import IndependenceTest


class RV(IndependenceTest):
    r"""
    Class for calculating the RV test statistic and p-value.

    RV is the multivariate generalization of the squared Pearson correlation
    coefficient [#1RV]_. The RV coefficient can be thought to be closely
    related to principal component analysis (PCA), canonical correlation
    analysis (CCA), multivariate regression, and statistical classification
    [#1RV]_.

    See Also
    --------
    CCA : CCA test statistic and p-value.

    Notes
    -----
    The statistic can be derived as follows [#1RV]_ [#2RV]_:

    Let :math:`x` and :math:`y` be :math:`(n, p)` samples of random variables
    :math:`X` and :math:`Y`. We can center :math:`x` and :math:`y` and then
    calculate the sample covariance matrix :math:`\hat{\Sigma}_{xy} = x^T y`
    and the variance matrices for :math:`x` and :math:`y` are defined
    similarly. Then, the RV test statistic is found by calculating

    .. math::

        \mathrm{RV}_n (x, y) =
            \frac{\mathrm{tr} \left( \hat{\Sigma}_{xy}
                                     \hat{\Sigma}_{yx} \right)}
            {\mathrm{tr} \left( \hat{\Sigma}_{xx}^2 \right)
             \mathrm{tr} \left( \hat{\Sigma}_{yy}^2 \right)}

    where :math:`\mathrm{tr} (\cdot)` is the trace operator.

    The p-value returned is calculated using a permutation test using a
    `permutation test <https://hyppo.neurodata.io/reference/tools.html#permutation-test>`_.

    References
    ----------
    .. [#1RV] Robert, P., & Escoufier, Y. (1976). A unifying tool for linear
              multivariate statistical methods: the RV‐coefficient. Journal
              of the Royal Statistical Society: Series C (Applied
              Statistics), 25(3), 257-265.
    .. [#2RV] Escoufier, Y. (1973). Le traitement des variables vectorielles.
              Biometrics, 751-760.
    """

    def __init__(self):
        IndependenceTest.__init__(self)

    def _statistic(self, x, y):
        r"""
        Helper function that calculates the RV test statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples and dimensions. That is, the shapes must be `(n, p)` where
            `n` is the number of samples and `p` is the number of dimensions.

        Returns
        -------
        stat : float
            The computed RV statistic.
        """
        centx = x - np.mean(x, axis=0)
        centy = y - np.mean(y, axis=0)

        # calculate covariance and variances for inputs
        covar = centx.T @ centy
        varx = centx.T @ centx
        vary = centy.T @ centy

        covar = np.trace(covar @ covar.T)
        stat = np.divide(
            covar, np.sqrt(np.trace(varx @ varx)) * np.sqrt(np.trace(vary @ vary))
        )
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=1):
        r"""
        Calculates the RV test statistic and p-value.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices. `x` and `y` must have the same number of
            samples and dimensions. That is, the shapes must be `(n, p)` where
            `n` is the number of samples and `p` is the number of dimensions.
        reps : int, optional (default: 1000)
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, optional (default: 1)
            The number of cores to parallelize the p-value computation over.
            Supply -1 to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed RV statistic.
        pvalue : float
            The computed RV p-value.

        Examples
        --------
        >>> import numpy as np
        >>> from hyppo.independence import RV
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = RV().test(x, y)
        >>> '%.1f, %.2f' % (stat, pvalue)
        '1.0, 0.00'

        The number of replications can give p-values with higher confidence
        (greater alpha levels).

        >>> import numpy as np
        >>> from hyppo.independence import RV
        >>> x = np.arange(7)
        >>> y = x
        >>> stat, pvalue = RV().test(x, y, reps=10000)
        >>> '%.1f, %.2f' % (stat, pvalue)
        '1.0, 0.00'
        """
        check_input = _CheckInputs(x, y, reps=reps)
        x, y = check_input()

        return super(RV, self).test(x, y, reps, workers, is_distsim=False)
