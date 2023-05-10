import numpy as np
from scipy.signal import correlate
from scipy.stats import chi2

from ._utils import _CheckInputs
from .base import TimeSeriesTest, TimeSeriesTestOutput


class LjungBox(TimeSeriesTest):
    r"""
    Ljung-Box for Cross Correlation (CorrX) test statistic and p-value.

    Parameters
    ----------
    max_lag : int, default: 0
        The maximum number of lags in the past to check dependence between ``x`` and the
        shifted ``y``. Also the ``M`` hyperparmeter below.

    Notes
    -----
    The statistic can be derived as follows
    :footcite:p:`mehtaIndependenceTestingMultivariate2020`:

    Let :math:`x` and :math:`y` be :math:`(n, 1)` and :math:`(n, 1)` series
    respectively, which each contain :math:`y` observations of the series
    :math:`(X_t)` and :math:`(Y_t)`. Similarly, let :math:`x[j:n]` be the
    :math:`(n-j, p)` last :math:`n-j` observations of :math:`x`. Let :math:`y[0:(n-j)]`
    be the :math:`(n-j, p)` first :math:`n-j` observations of :math:`y`. Let :math:`M`
    be the maximum lag hyperparameter. The cross distance correlation is,

    .. math::

        \mathrm{DcorrX}_n (x, y) =  \sum_{j=0}^M \frac{n-j}{n}
                                    Dcorr_n (x[j:n], y[0:(n-j)])

    The p-value returned is calculated using a permutation test.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, max_lag=0):
        self.is_distance = False
        TimeSeriesTest.__init__(self, compute_distance=None, max_lag=max_lag)

    def statistic(self, x, y):
        r"""
        Helper function that calculates the Ljung-Box cross correlation test statistic.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            samples. That is, the shapes must be ``(n, 1)`` and ``(n, 1)`` where
            `n` is the number of samples.

        Returns
        -------
        stat : float
            The computed Ljung-Box statistic.
        opt_lag : int
            The computed optimal lag.
        """
        n = x.shape[0]

        ccf = correlate(x - x.mean(), y - y.mean())[n - 1 :].ravel()
        ccf /= np.arange(n, 0, -1)
        ccf /= np.std(x) * np.std(y)
        stat = ccf[1 : self.max_lag + 1] ** 2 / (n - np.arange(1, self.max_lag + 1))

        self.stat = np.sum(stat)
        self.opt_lag = np.argmax(stat) + 1

        return self.stat, self.opt_lag

    def test(self, x, y, reps=1000, workers=1, auto=True, random_state=None):
        check_input = _CheckInputs(
            x,
            y,
            max_lag=self.max_lag,
        )
        x, y, self.max_lag = check_input()

        # calculate observed test statistic
        stat, opt_lag = self.statistic(x, y)

        if auto and x.shape[0] > 20:
            pvalue = chi2.sf(stat, self.max_lag)
            self.null_dist = None
        elif auto is False:
            stat, pvalue, stat_list = super(LjungBox, self).test(
                x=x, y=y, reps=reps, workers=workers, is_distsim=False, random_state=random_state
            )

        self.pvalue = pvalue
        test_dict = {"opt_lag": opt_lag}

        return TimeSeriesTestOutput(stat, pvalue, test_dict)
