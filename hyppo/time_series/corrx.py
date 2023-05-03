from ._utils import _CheckInputs
from .base import TimeSeriesTest, TimeSeriesTestOutput, _perm_stat

from joblib import Parallel, delayed
from sklearn.utils import check_random_state
import numpy as np
from scipy.stats import chi2


class LjungBox(TimeSeriesTest):
    r"""
    Ljung-Box Cross Correlation (CorrX) test statistic and p-value.

    Parameters
    ----------
    max_lag : int, default: 0
        The maximum number of lags in the past to check dependence between ``x`` and the
        shifted ``y``. Also the ``M`` hyperparmeter below.

    Notes
    -----
    The statistic can be derived as follows
    :footcite:p:`mehtaIndependenceTestingMultivariate2020`:

    Let :math:`x` and :math:`y` be :math:`(n, p)` and :math:`(n, q)` series
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
        TimeSeriesTest.__init__(self, compute_distance=None, max_lag=max_lag)

    def statistic(self, x, y):
        stat = _statistic(x, y, self.max_lag)
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=1, auto=True, random_state=None):
        check_input = _CheckInputs(
            x,
            y,
            max_lag=self.max_lag,
        )
        x, y = check_input()

        # calculate observed test statistic
        stat = self.statistic(x, y)

        if auto and x.shape[0] > 20:
            chi2.sf(stat, self.max_lag)

        # make RandomState seeded array
        if random_state is not None:
            rng = check_random_state(random_state)
            random_state = rng.randint(np.iinfo(np.int32).max, size=reps)

        # make random array
        else:
            random_state = np.random.randint(np.iinfo(np.int32).max, size=reps)

        # calculate null distribution
        null_dist = np.array(
            Parallel(n_jobs=workers)(
                [delayed(_perm_stat)(self.statistic, x, y, rng) for rng in random_state]
            )
        )
        pvalue = (1 + (null_dist >= stat).sum()) / (1 + reps)
        self.pvalue = pvalue
        self.null_dist = null_dist

        return TimeSeriesTestOutput(stat, pvalue, null_dist)


def _statistic():
    pass
