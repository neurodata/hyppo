import numpy as np

from .base import RandomForestTest
from ._utils import _CheckInputs, rf_xmat
from ..independence import Dcorr
from .._utils import euclidean


class DcorrRF(RandomForestTest):
    r"""
    Class for calculating the random forest based Dcorr test statistic and p-value.
    """

    def __init__(self, ntrees=100):
        self.ntrees = ntrees
        RandomForestTest.__init__(self)

    def _statistic(self, x, y):
        r"""
        Helper function that calculates the random forest based Dcorr test statistic.

        y must be categorical
        """
        simx = rf_xmat(x, y, self.ntrees)
        simy = euclidean(y)
        stat = Dcorr(compute_distance=None)._statistic(simx, simy)
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=1, random_state=None):
        r"""
        Calculates the random forest based Dcorr test statistic and p-value.
        """
        check_input = _CheckInputs(x, y, reps=reps, ntrees=self.ntrees)
        x, y = check_input()

        return super(DcorrRF, self).test(x, y, reps, workers, random_state)
