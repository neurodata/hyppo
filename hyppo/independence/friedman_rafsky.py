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

    def statistic(self, x, y, algorithm='Kruskal'):
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
        y = np.transpose(y)

        MST_connections = MST(x, algorithm)
        stat = num_runs(y, MST_connections)

        self.stat = stat

        return stat

    def test(
        self,
        x,
        y,
        algorithm="Kruskal",
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
        algoritm : str, default: 'Kruskal'
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


class Graph:
    r"""
    Helper class to find the MST for a given dataset using
    Kruskal's MST algorithm. Takes in node lables and edge weights
    before returning all pairs of connected node labels in the
    resultant MST.
    """

    def __init__(self, vertex):

        self.V = vertex
        self.graph = []

    def add_edge(self, v1, v2, w):
        r"""Helper function to add edge to the parent graph

        Parameters
        ----------
        v1, v2 : int
            Input node indeces v1, v2 to be connected by edge in
            the graph.
        weight : float
            The Euclidean distance between these two points.
        """
        self.graph.append([v1, v2, w])

    def search(self, parent, i):
        r"""
        Method for determining location of vertex in existing tree
        """
        if parent[i] == i:
            return i

        return self.search(parent, parent[i])

    def apply_union(self, parent, rank, x, y):
        r"""
        Method for determining if current node already exists in
        minimum spanning tree and severing all edges deemed inefficient.
        """
        xroot = self.search(parent, x)
        yroot = self.search(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskal(self):
        r"""
        Helper function to calculate minimum spanning tree
        via Kruskal's algorithm.
        """
        result = []
        i, e = 0, 0
        self.graph = sorted(self.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self.V):
            parent.append(node)
            rank.append(0)
        while e < self.V - 1:
            v1, v2, w = self.graph[i]
            i = i + 1
            x = self.search(parent, v1)
            y = self.search(parent, v2)
            if x != y:
                e = e + 1
                result.append([v1, v2])
                self.apply_union(parent, rank, x, y)

        return result


@jit(nopython=True, cache=True)
def prim(weight_mat):
    r"""
    Helper function to read weighted matrix input and compute minimum
    spanning tree via Prim's algorithm.

    Parameters
    ----------
    data : ndarry
        Weighted connection matrix.

    Returns
    -------
    MST_connections : list
        List of pairs of nodes connected in final MST.
    """
    INF = 9999999
    V = len(y)
    selected = np.zeros(len(y))
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
                        if minimum > G[i][j]:
                            minimum = G[i][j]
                            x = i
                            y = j
        MST_connections.append([x, y])
        selected[y] = True
        no_edge += 1

    return MST_connections


@jit(nopython=True, cache=True)
def MST(x, algorithm):
    r"""
    Helper function to read input data and calculate Euclidean distance
    between each possible pair of points before finding MST.

    Parameters

    ----------
    data : ndarry
        Dataset such that each column corresponds to a point of data.
    algoritm : str, default: 'Kruskal'
            The algorithm to be used to determine the minimum spanning tree.
            Currently only 'Kruskal' and 'Prim' are supported.

    Returns
    -------
    MST_connections : list
        List of pairs of nodes connected in final MST.
    """
    if algorithm == "Kruskal":

        g = self.Graph(len(x[0]))

        for i in range(len(x[0])):

            for j in range(i + 1, len(x[0])):
                weight = np.linalg.norm(x[:, i] - x[:, j])
                g.add_edge(i, j, weight)

        MST_connections = g.kruskal()

    if algorithm == "Prim":

        G = zeros((len(x[0]), len(x[0])))

        for i in range(len(x[0])):

            for j in range(i + 1, len(x[0])):
                weight = np.linalg.norm(x[:, i] - x[:, j])
                G[i][j] = weight
                G[j][i] = weight

        MST_connections = prim(G)

    return MST_connections


@jit(nopython=True, cache=True)
def num_runs(y, MST_connections):
    r"""
    Helper function to determine number of independent
    'runs' from MST connections.

    Parameters
    ----------
    y : ndarry
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
