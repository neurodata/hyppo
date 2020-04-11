import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import RandomForestTest
from ._utils import _CheckInputs, sim_matrix
from ..independence import Dcorr
from .._utils import euclidean, perm_test


class MGCRF(RandomForestTest):
    r"""
    Class for calculating the random forest based Dcorr test statistic and p-value.
    """

    def __init__(self, clf=RandomForestRegressor(n_estimators=500)):
        self.clf = clf
        self.first_time = True
        RandomForestTest.__init__(self)

    def _statistic(self, x, y):
        r"""
        Helper function that calculates the random forest based Dcorr test statistic.

        y must be categorical
        """
        if self.first_time:
            y = y.reshape(-1)
            self.clf.fit(x, y)
            self.first_time = False
        distx = np.sqrt(1 - sim_matrix(self.clf, x))
        y = y.reshape(-1, 1)
        disty = euclidean(y)
        stat = Dcorr(compute_distance=None)._statistic(distx, disty)
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=1):
        r"""
        Calculates the random forest based Dcorr test statistic and p-value.
        """
        check_input = _CheckInputs(x, y, reps=reps)
        x, y = check_input()

        stat, pvalue = perm_test(self._statistic, x, y, reps=reps, workers=workers)
        self.stat = stat
        self.pvalue = pvalue

        return stat, pvalue