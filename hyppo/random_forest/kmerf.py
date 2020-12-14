import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .base import RandomForestTest
from ._utils import _CheckInputs, sim_matrix
from ..independence import Dcorr
from .._utils import euclidean, perm_test


FOREST_TYPES = {
    "classifier" : RandomForestClassifier,
    "regressor" : RandomForestRegressor
}


class KMERF(RandomForestTest):
    r"""
    Class for calculating the random forest based Dcorr test statistic and p-value.
    """

    def __init__(self, forest="regressor", ntrees=500, **kwargs):
        if forest in FOREST_TYPES.keys():
            self.clf = FOREST_TYPES[forest](n_estimators=ntrees, **kwargs)
        else:
            raise ValueError("forest must be classifier or regressor")
        RandomForestTest.__init__(self)

    def _statistic(self, x, y):
        r"""
        Helper function that calculates the random forest based Dcorr test statistic.

        y must be categorical
        """
        y = y.reshape(-1)
        self.clf.fit(x, y)
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

        stat, pvalue = perm_test(self._statistic, x, y, reps=reps, workers=workers, is_distsim=False)
        self.stat = stat
        self.pvalue = pvalue

        return stat, pvalue
