import numpy as np
from numba import jit
from sklearn.metrics.pairwise import pairwise_kernels

from ..tools import check_perm_blocks_dim, compute_dist

from scipy.spatial.distance import pdist, squareform

# from ._utils import _CheckInputs
from .base import ConditionalIndependenceTest, ConditionalIndependenceTestOutput


class CDcorr(ConditionalIndependenceTest):
    r"""
    Conditional Distance Correlation (CDcorr) test statistic and p-value.


    Reference
    ---------
    D.W. Scott, “Multivariate Density Estimation: Theory, Practice, and Visualization”, John Wiley & Sons, New York, Chicester, 1992.
    """

    def __init__(self, compute_distance="euclidean", bandwidth=None, **kwargs):
        r"""
        bandwith :
        """
        self.compute_distance = compute_distance
        self.bandwidth = bandwidth

        # set is_distance to true if compute_distance is None
        self.is_distance = False
        if not compute_distance:
            self.is_distance = True
        ConditionalIndependenceTest.__init__(self)

    def compute_kern(self, data):
        n, d = data.shape

        if self.bandwidth is None:
            # Assumes independent variables
            # Scott's rule of thumb
            factor = np.power(n, (-1.0 / (d + 4)))
            stds = np.std(data, axis=0, ddof=1)
            bandwidth_ = factor * stds
        elif isinstance(self.bandwidth, (int, float)):
            bandwidth_ = np.repeat(self.bandwidth, d)
        else:
            bandwidth_ = self.bandwidth

        # Compute constants
        denom = np.power(2 * np.pi, d / 2) * np.power(bandwidth_.prod(), 0.5)

        sim_mat = pdist(data, "sqeuclidean", w=1 / bandwidth_**2)
        sim_mat = squareform(-0.5 * sim_mat)
        np.exp(sim_mat, sim_mat)
        sim_mat /= denom

        return sim_mat

    def statistic(self, x, y, z):
        distx = x
        disty = y
        distz = z

        if not self.is_distance:
            distx, disty = compute_dist(
                x,
                y,
                metric=self.compute_distance,
            )
            distz = self.compute_kern(z)

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

    # cdcov *= 12 * np.power(distz.mean(axis=0), 4)

    return cdcov


def _cdcorr(distx, disty, distz):
    varx = _cdcov(distx, distx, distz)
    vary = _cdcov(disty, disty, distz)
    covar = _cdcov(distx, disty, distz)

    # stat is 0 with negative variances (would make denominator undefined)
    denom = varx * vary
    non_positives = denom <= 0
    if np.any(non_positives):
        denom[non_positives] = 0
    stat = np.divide(
        covar, np.sqrt(denom), out=np.zeros_like(covar), where=np.invert(non_positives)
    )

    return stat
