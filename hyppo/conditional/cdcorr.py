import numpy as np
from numba import jit
from sklearn.metrics.pairwise import pairwise_kernels

from ..tools import check_perm_blocks_dim, compute_dist

# from ._utils import _CheckInputs
from .base import ConditionalIndependenceTest, ConditionalIndependenceTestOutput


class CDcorr(ConditionalIndependenceTest):
    r"""
    Conditional Distance Correlation (CDcorr) test statistic and p-value.

    """

    def __init__(self, compute_distance="euclidean", bandwith=None, **kwargs):
        r"""
        bandwith :
        """
        self.compute_distance = compute_distance
        self.bandwith = bandwith

        # set is_distance to true if compute_distance is None
        self.is_distance = False
        if not compute_distance:
            self.is_distance = True
        ConditionalIndependenceTest.__init__(self)

    def compute_kern(self, data):
        d = data.shape[1]

        # Compute constants
        denom = np.power(2 * np.pi, d / 2.0) * np.power(self.bandwith, d / 2)
        constant = 1 / denom
        gamma = 1 / (self.bandwith * 2)

        kern = pairwise_kernels(data, metric="rbf", gamma=gamma) * constant
        return kern

    def statistic(self, x, y, z):
        distx = x
        disty = y
        distz = z

        if not self.is_distance:
            distx, disty = compute_dist(
                x, y, metric=self.compute_distance, **self.kwargs
            )
            distz = self.compute_kern()

        stat = _cdcorr(distx, disty, distz)
        self.stat = stat

        return stat

    def test(
        self,
        x,
        y,
        z,
        reps=1000,
        workers=1,
        auto=True,
        perm_blocks=None,
        random_state=None,
    ):

        return ConditionalIndependenceTestOutput(stat, pvalue)


def _weighted_center_distmat(distx, weights):
    n = distx.shape[0]

    scl = np.sum(weights)
    row_sum = np.sum(np.multiply(distx, weights), axis=1) / scl
    total_sum = weights @ row_sum / scl

    cent_distx = distx - row_sum.reshape(-1, n).T - row_sum.reshape(-1, n) + total_sum

    return cent_distx


def _cdcov(distx, disty, distz):
    n = distx.shape[0]

    cdcov = np.zeros(n)

    for i in range(n):
        r = distz[[i]]
        cdx = _weighted_center_distmat(distx, distz[i])
        cdy = _weighted_center_distmat(disty, distz[i])
        cdcov[i] = (cdx * cdy * r * r.T).sum() / r.sum() ** 2

    cdcov *= 12 * np.power(distz.mean(axis=0), 4)

    return cdcov.mean()


def _cdcorr(distx, disty, distz):
    varx = _cdcov(distx, distx, distz)
    vary = _cdcov(disty, disty, distz)
    covar = _cdcov(distx, disty, distz)

    # stat is 0 with negative variances (would make denominator undefined)
    if varx <= 0 or vary <= 0:
        stat = 0

    # calculate generalized test statistic
    else:
        stat = covar / np.real(np.sqrt(varx * vary))

    return stat
