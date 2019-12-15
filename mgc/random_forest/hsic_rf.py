import numpy as np
from numba import njit

from .base import RandomForestTest
from ._utils import _CheckInputs, rf_xmat
from ..independence import Hsic
from .._utils import gaussian


class HsicRF(RandomForestTest):
    r"""
    Class for calculating the random forest based Hsic test statistic and p-value.
    """

    def __init__(self, ntrees=500):
        self.ntrees = ntrees
        RandomForestTest.__init__(self)

    def _statistic(self, x, y):
        r"""
        Helper function that calculates the random forest based Hsic test statistic.
        """
        simx = rf_xmat(x, y, self.ntrees)
        simy = gaussian(y)
        stat = Hsic(compute_kernel=None)._statistic(simx, simy)
        self.stat = stat

        return stat

    def test(self, x, y, reps=1000, workers=1, random_state=None):
        r"""
        Calculates the random forest based Hsic test statistic and p-value.
        """
        check_input = _CheckInputs(x, y, reps=reps, ntrees=self.ntrees)
        x, y = check_input()

        return super(HsicRF, self).test(x, y, reps, workers, random_state)
