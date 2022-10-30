import random
from typing import NamedTuple

from numba import jit
import numpy as np

from .base import IndependenceTest
from ..tools import perm_test


class FRTestOutput(NamedTuple):
    stat: float
    pvalue: float
    uncor_stat: dict


class FriedmanRafsky(IndependenceTest):
    r"""
    Friedman-Rafksy (FR) test statistic and p-value.
    This is a multivariate extension of the Wald-Wolfowitz
    runs test for randomness. The normal concept of a 'run'
    is replaced by a minimum spanning tree (MST) calculated between
    the points in respective data sets with edge weights defined
    as the Euclidean distance between two such points. After MST
    has been determined, all edges such that both corresponding
    nodes do not belong to the same class are severed and the
    number of independent resulting trees is counted. This test is
    consistent against similar tests.

    Notes
    -----
    The statistic can be derived as follows
    :footcite:p:`friedmanMultivariateGeneralizationsoftheWaldWolfowitzandSmirnovTwoSampleTests1979`

    Let :math:`x` be a combined sample of :math:`(n, p)` and :math:`(m, p)`
    samples of random variables :math:`X` and let :math:`y` be a :math:`(n+m, 1)`
    array of labels :math:`Y`. We can then create a graph such that each point in
    :math:`X` is connected to each other point in :math:`X` by an edge weighted by
    the euclidean distance inbetween those points. The minimum spanning tree is then
    calculated and all edges such that the labels in :math:`Y` are not from the same
    class are removed. The number of independent graphs is then summed to determine
    the uncorrected statistic for the test.

    The p-value and null distribution for the corrected statistic are calculated via
    a permutation test using :meth:`hyppo.tools.perm_test`.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, **kwargs):

        IndependenceTest.__init__(self, **kwargs)

    def statistic(self, x, y):
        r"""
        Helper function that calculates the Friedman Rafksy test statistic.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            rows. That is, the shapes must be ``(n, p)`` and ``(n, 1)`` where
            `n` is the number of combined samples and `p` is the number of
            dimensions. ``y`` is the array of labels corresponding to the two
            samples, respectively.

        Returns
        -------
        stat : float
            The computed (uncorrected) Friedman Rafsky statistic. A value between
            ``2`` and ``n``.
        """
        x = np.transpose(x)
        labels = np.transpose(y)

        MST_connections = MST(x, labels)
        stat = _num_runs(labels, MST_connections)

        return stat

    def test(
        self,
        x,
        y,
        reps=1000,
        workers=1,
        random_state=None,
    ):
        r"""
        Calculates the Friedman Rafsky test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray of float
            Input data matrices. ``x`` and ``y`` must have the same number of
            rows. That is, the shapes must be ``(n, p)`` and ``(n, 1)`` where
            `n` is the number of combined samples and `p` is the number of
            dimensions. ``y`` is the array of labels corresponding to the two
            samples, respectively.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.
        random_state : int, default: None
            The random_state for permutation testing to be fixed for
            reproducibility.

        Returns
        -------
        stat : float
            The computed (corrected) Friedman Rafsky statistic.
        pvalue : float
            The computed Friedman Rafsky p-value.
        uncor_stat : float
            The computed (uncorrected) Friedman Rafsky statistic.
        """
        uncor_stat, pvalue, null_dist = perm_test(
            self.statistic,
            x,
            y,
            reps,
            workers,
            is_distsim=False,
            random_state=random_state,
        )
        self.uncor_stat = uncor_stat
        stat = (uncor_stat - np.mean(null_dist)) / np.std(null_dist)
        self.stat = stat

        return FRTestOutput(stat, pvalue, uncor_stat)


def _num_runs(labels, MST_connections):
    r"""
    Helper function to determine number of independent
    'runs' from MST connections.

    Parameters
    ----------
    labels : ndarry of float
        Lables corresponding to respective classes of samples.
    MST_connections: list of int
        List containing pairs of points connected in final MST.

    Returns
    -------
    run_count : int
        Number of runs after severing all such edges with nodes of
        differing class labels.
    """
    run_count = 1

    for x in MST_connections:
        if labels[x[0]] != labels[x[1]]:
            run_count += 1

    return run_count


@jit(nopython=True, cache=True)  # pragma: no cover
def prim(weight_mat, labels):
    r"""
    Helper function to read weighted matrix input and compute minimum
    spanning tree via Prim's algorithm.

    Parameters
    ----------
    weight_mat : ndarry of float
        Weighted connection matrix.
    labels : ndarry of int
        Lables corresponding to respective classes of samples.

    Returns
    -------
    MST_connections : list of int
        List of pairs of nodes connected in final MST.
    """
    INF = 9999999
    V = len(labels)
    selected = np.zeros(len(labels))
    no_edge = 0
    selected[0] = True
    MST_connections = []

    while no_edge < V - 1:
        minimum = INF
        x = 0
        y = 0
        for i in range(V):
            if selected[i]:
                for j in range(V):
                    if (not selected[j]) and weight_mat[i][j]:
                        # not in selected and there is an edge
                        if minimum > weight_mat[i][j]:
                            minimum = weight_mat[i][j]
                            x = i
                            y = j
        MST_connections.append([x, y])
        selected[y] = True
        no_edge += 1

    return MST_connections


@jit(nopython=True, cache=True)
def MST(x, labels):  # pragma: no cover
    r"""
    Helper function to read input data and calculate Euclidean distance
    between each possible pair of points before finding MST.

    Parameters
    ----------
    x : ndarry of float
        Dataset such that each column corresponds to a point of data.
    labels : ndarry of int
        Lables corresponding to respective classes of samples.

    Returns
    -------
    MST_connections : list
        List of pairs of nodes connected in final MST.
    """

    G = np.zeros((len(x[0]), len(x[0])))

    for i in range(len(x[0])):

        for j in range(i + 1, len(x[0])):
            weight = np.linalg.norm(x[:, i] - x[:, j])
            G[i][j] = weight
            G[j][i] = weight

    MST_connections = prim(G, labels)

    return MST_connections
