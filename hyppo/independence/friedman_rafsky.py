import random
from numba import jit
import numpy as np

from .base import IndependenceTest, IndependenceTestOutput
from ..tools import perm_test


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
    """

    def __init__(self, **kwargs):

        IndependenceTest.__init__(self, **kwargs)

    def statistic(self, x, y, algorithm="Prim"):
        r"""
        Helper function that calculates the Friedman Rafksy test statistic.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            rows. That is, the shapes must be ``(n, p)`` and ``(n, 1)`` where
            `n` is the number of combined samples and `p` is the number of
            dimensions. ``y`` is the array of labels corresponding to the two
            samples, respectively.
        algoritm : str, default: 'Kruskal'
            The algorithm to be used to determine the minimum spanning tree.
            Currently only 'Kruskal' and 'Prim' are supported.

        Returns
        -------
        stat : float
            The computed Friedman Rafsky statistic.
        """
        x = np.transpose(x)
        labels = np.transpose(y)

        MST_connections = MST(x, labels, algorithm)
        stat = num_runs(labels, MST_connections)

        self.stat = stat

        return stat

    def test(
        self,
        x,
        y,
        algorithm="Prim",
        reps=1000,
        workers=1,
        is_distsim=False,
        perm_blocks=None,
        random_state=None,
    ):
        r"""
        Calculates the Friedman Rafsky test statistic and p-value.

        Parameters
        ----------
        x,y : ndarray
            Input data matrices. ``x`` and ``y`` must have the same number of
            rows. That is, the shapes must be ``(n, p)`` and ``(n, 1)`` where
            `n` is the number of combined samples and `p` is the number of
            dimensions. ``y`` is the array of labels corresponding to the two
            samples, respectively.
        algoritm : str, default: 'Prim'
            The algorithm to be used to determine the minimum spanning tree.
            Currently only 'Kruskal' and 'Prim' are supported.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.
        perm_blocks : None or ndarray, default: None
            Defines blocks of exchangeable samples during the permutation test.
            If None, all samples can be permuted with one another. Requires `n`
            rows. At each column, samples with matching column value are
            recursively partitioned into blocks of samples. Within each final
            block, samples are exchangeable. Blocks of samples from the same
            partition are also exchangeable between one another. If a column
            value is negative, that block is fixed and cannot be exchanged.
        random_state : int, default: None
            The random_state for permutation testing to be fixed for
            reproducibility.

        Returns
        -------
        stat : float
            The computed Friedman Rafsky statistic.
        pvalue : float
            The computed Friedman Rafsky p-value.
        null_dist : array
            The null distribution of Friedman Rafsky test statistics.
        """

        stat, pvalue, null_dist = perm_test(
            self.statistic, x, y, reps, workers, is_distsim, perm_blocks, random_state
        )

        return IndependenceTestOutput(stat, pvalue, null_dist)


@jit(nopython=True, cache=True)
def prim(weight_mat, labels):
    r"""
    Helper function to read weighted matrix input and compute minimum
    spanning tree via Prim's algorithm.

    Parameters
    ----------
    weight_mat : ndarry
        Weighted connection matrix.
    labels : ndarry
        Lables corresponding to respective classes of samples.

    Returns
    -------
    MST_connections : list
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
                    if (not selected[j]) and G[i][j]:
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
def MST(x, labels, algorithm):
    r"""
    Helper function to read input data and calculate Euclidean distance
    between each possible pair of points before finding MST.

    Parameters

    ----------
    data : ndarry
        Dataset such that each column corresponds to a point of data.
    labels : ndarry
        Lables corresponding to respective classes of samples.
    algoritm : str, default: 'Kruskal'
            The algorithm to be used to determine the minimum spanning tree.
            Currently only 'Kruskal' and 'Prim' are supported.

    Returns
    -------
    MST_connections : list
        List of pairs of nodes connected in final MST.
    """

    if algorithm == "Prim":

        G = np.zeros((len(x[0]), len(x[0])))

        for i in range(len(x[0])):

            for j in range(i + 1, len(x[0])):
                weight = np.linalg.norm(x[:, i] - x[:, j])
                G[i][j] = weight
                G[j][i] = weight

        MST_connections = prim(G, labels)

    return MST_connections


@jit(nopython=True, cache=True)
def num_runs(labels, MST_connections):
    r"""
    Helper function to determine number of independent
    'runs' from MST connections.

    Parameters
    ----------
    labels : ndarry
        Lables corresponding to respective classes of samples.
    MST_connections: list
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
