from functools import partial

import numpy as np
from scipy.spatial.distance import pdist, squareform

from ..tools import compute_dist, perm_test
from ._utils import _CheckInputs
from .base import ConditionalIndependenceTest, ConditionalIndependenceTestOutput


class CDcov(ConditionalIndependenceTest):
    r"""
    Conditional Distance Covariance (CDcorr) test statistic and p-value.


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

        ConditionalIndependenceTest.__init__(self, **kwargs)

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
            distz, bandwidth_ = _compute_kern(z, self.bandwidth)

        stat = _cdcov(distx, disty, distz).mean()
        self.stat = stat

        return stat

    def test(
        self,
        x,
        y,
        z,
        reps=1000,
        workers=1,
        random_state=None,
    ):
        check_input = _CheckInputs(
            x,
            y,
            z,
            reps=reps,
        )
        x, y, z = check_input()

        if not self.is_distance:
            x, y = compute_dist(x, y, metric=self.compute_distance, **self.kwargs)
            z = self._compute_kern(z)

            self.is_distance = True

        stat, pvalue, null_dist = perm_test(
            self.statistic,
            x,
            y,
            z,
            reps=reps,
            workers=workers,
            is_distsim=self.is_distance,
            random_state=random_state,
            permuter=partial(self._permuter, probs=z),
        )
        self.stat = stat
        self.pvalue = pvalue
        self.null_dist = null_dist

        return ConditionalIndependenceTestOutput(stat, pvalue)


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

        ConditionalIndependenceTest.__init__(self, **kwargs)

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
            distz, bandwidth_ = _compute_kern(z, self.bandwidth)

        stat = _cdcorr(distx, disty, distz).mean()
        self.stat = stat

        return stat

    def test(
        self,
        x,
        y,
        z,
        reps=1000,
        workers=1,
        random_state=None,
    ):
        check_input = _CheckInputs(
            x,
            y,
            z,
            reps=reps,
        )
        x, y, z = check_input()

        if not self.is_distance:
            x, y = compute_dist(x, y, metric=self.compute_distance, **self.kwargs)
            z = self._compute_kern(z)

            self.is_distance = True

        stat, pvalue, null_dist = perm_test(
            self.statistic,
            x,
            y,
            z,
            reps=reps,
            workers=workers,
            is_distsim=self.is_distance,
            random_state=random_state,
            permuter=partial(_permuter, probs=z),
        )
        self.stat = stat
        self.pvalue = pvalue
        self.null_dist = null_dist

        return ConditionalIndependenceTestOutput(stat, pvalue)

    def _permuter(self, probs, rng, axis=1):
        """
        Weighted sampling with replacement
        """
        n = probs.shape[1 - axis]
        sums = probs.sum(axis=1, keepdims=True)
        idx = ((probs / sums).cumsum(axis=1) > rng.rand(n)[:, None]).argmax(axis=1)

        return idx


def _compute_kern(data, bandwidth):
    n, d = data.shape

    if bandwidth is None:
        # Assumes independent variables
        # Scott's rule of thumb
        factor = np.power(n, (-1.0 / (d + 4)))
        stds = np.std(data, axis=0, ddof=1)
        bandwidth_ = factor * stds

        # TODO Implement silverman's rule of thumb
    elif isinstance(bandwidth, (int, float)):
        bandwidth_ = np.repeat(bandwidth, d)
    else:
        bandwidth_ = bandwidth

    # Compute constants
    denom = np.power(2 * np.pi, d / 2) * np.power(bandwidth_.prod(), 0.5)

    sim_mat = pdist(data, "sqeuclidean", w=1 / bandwidth_**2)  # Section 4.2 equation
    sim_mat = squareform(-0.5 * sim_mat)
    np.exp(sim_mat, sim_mat)
    sim_mat /= denom

    return sim_mat, bandwidth


def _permuter(probs, rng, axis=1):
    """
    Weighted sampling with replacement
    """
    n = probs.shape[1 - axis]
    sums = probs.sum(axis=1, keepdims=True)
    idx = ((probs / sums).cumsum(axis=1) > rng.rand(n)[:, None]).argmax(axis=1)

    return idx


def _weighted_center_distmat(distx, weights):
    """Centers the distance matrices"""

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
