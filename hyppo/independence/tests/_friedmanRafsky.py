#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy
import random
from numba import jit

from numpy import mean, zeros, linalg.norm, std

from .base import IndependenceTest, IndependenceTestOutput


class FriedmanRafsky(IndependenceTest):
    
    
    r"""
    Friedman-Rafksy (FR) test statistic and p-value.
    
    This is a multivariate extension of the Wald-Wolfowitz 
    runs test for randomness. The normal concept of a 'run'
    is replaced by a minimum spanning tree calculated between
    the points in respective data sets with edge weights defined
    as the Euclidean distance between two such points. After MST
    has been determined, all edges such that both corresponding
    nodes do not belong to the same class are severed and the
    number of independent resulting trees is counted. This test is 
    consistent against similar tests 
    :footcite:p:`GSARMultivariateAdaptationofWaldWolfowitzTest`
    
    """
    
    def __init__(self, **kwargs):
        
        IndependenceTest.__init__(self, **kwargs)
    
    
    
    def pval(self, perm_stat, true_stat):
        
        r"""
        Helper function that calculates the Friedman Rafksy p-value.
        
        """
        
        pvalue = (sum(perm_stat <= true_stat) + 1) / (len(perm_stat) + 1)
        
        return pvalue
    
    
    
    def test(self, data, labels, perm, algorithm):
        
        r"""
        Function to take in data, labels, nperm and return both test-statistic and
        p-value.
        
        Parameters
        ----------
        
        data, labels, nperm : ndarray, ndarray, int
            Data is array such that each column corresponds to a given point. Labels
            are the corresponding data labels for each point. It is of note that
            data and labels must have the same number of columns. Nperm is the
            number of randomized iterations to be used in calculating the test_statistic
            and p-value.
            
        Return
        ------
        
        W_true, p_value : float, float
            Corresponding test-statistic and p-value for the given data, labels, and
            number of randomized permutations used.
            
        """
        
        num_rows, num_cols = data.shape
        
        if num_cols != len(labels):
            raise IndexError('Number of features and labels not equal')
        
        
        MST_connections = MST(data, algorithm)
        
        runs_true = num_runs(labels, MST_connections)
        
        runs = permutation(perm, labels, MST_connections)
        
        W_perm = ((runs - mean(runs)) / std(runs))

        stat = (runs_true - mu_runs) / sd_runs
        
        self.stat = stat
        
        pvalue = pval(W_perm, stat)
        
        return IndependenceTestOutput(stat, pvalue)
    
    
@jit(nopython=True, cache=True)
class Graph:

    r"""
    Helper class that finds the MST for a given dataset using
    Kruskal's MST algorithm. Takes in node lables and edge weights
    before returning all pairs of connected node labels in the 
    resultant MST.

    Parameters
    ----------

    i,j,weight : int, int, float
        Input node indeces i, j, and corresponding Euclidean distance
        edge weight between them. 

    Returns
    -------

    result : list
        List of pairs of nodes connected in final MST.

    """

    def __init__(self2, vertex):

        self2.V = vertex
        self2.graph = [] #Empty matrix for holding vertices and weights connecting them


    def add_edge(self2, v1, v2, w):

        self2.graph.append([v1, v2, w]) #Add method for creating edges between vertices


    def search(self2, parent, i): #Method for determining location of vertex in existing tree

        if parent[i] == i:
            return i

        return self2.search(parent, parent[i])


    def apply_union(self2, parent, rank, x, y): #Method for deleting and merging branches

        xroot = self2.search(parent, x)
        yroot = self2.search(parent, y)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1


    def kruskal(self2):

        result = []
        i, e = 0, 0
        self2.graph = sorted(self2.graph, key=lambda item: item[2])
        parent = []
        rank = []
        for node in range(self2.V):
            parent.append(node)
            rank.append(0)
        while e < self2.V - 1:
            v1, v2, w = self2.graph[i]
            i = i + 1
            x = self2.search(parent, v1)
            y = self2.search(parent, v2)
            if x != y:
                e = e + 1
                result.append([v1, v2])
                self2.apply_union(parent, rank, x, y)

        return result
    

@jit(nopython=True, cache=True)
def prim(weight_mat):
        
    r"""
    Helper function to read weighted matrix input and compute minimum
    spanning tree via Prim's algorithm

    Parameters
    ----------

    data : ndarry
        Weighted connection matrix

    Returns
    -------
    MST_connections : list
        List of pairs of nodes connected in final MST.

    """

    INF = 9999999

    V = len(labels)

    selected = zeros(len(labels))

    no_edge = 0

    selected[0] = True

    MST_connections = []

    while (no_edge < V - 1):

        minimum = INF
        x = 0
        y = 0
        for i in range(V):
            if selected[i]:
                for j in range(V):
                    if ((not selected[j]) and G[i][j]):  
                        # not in selected and there is an edge
                        if minimum > G[i][j]:
                            minimum = G[i][j]
                            x = i
                            y = j
        MST_connections.append([x,y])
        selected[y] = True
        no_edge += 1

    return MST_connections


@jit(nopython=True, cache=True)
def MST(data, algorithm):
        
    r"""
    Helper function to read input data and calculate Euclidean distance
    between each possible pair of points before finding MST.

    Parameters
    ----------

    data : ndarry
        Dataset such that each column corresponds to a point of data.

    Returns
    -------
    MST_connections : list
        List of pairs of nodes connected in final MST.

    """
    if algorithm == 'Kruskal':

        g = Graph(len(data[0]-1))

        for i in range(len(data[0])):
            j = i + 1

            while j <= (len(data[0]) - 1):
                weight = linalg.norm(data[:,i] - data[:,j])
                g.add_edge(i, j, weight)
                j += 1;

        MST_connections = g.kruskal()

    if algorithm == 'Prim':

        G = zeros((len(data[0]), len(data[0])))

        for i in range(len(data[0])):
            j = i + 1

        while j <= (len(data[0]) - 1):
            weight = linalg.norm(data[:,i] - data[:,j])
            G[i][j] = weight
            G[j][i] = weight
            j += 1;

        MST_connections = prim(G)

    return MST_connections


@jit(nopython=True, cache=True)
def num_runs(labels, MST_connections):
        
    r"""
    Helper function to determine number of independent
    'runs' from MST connections

    Parameters
    ----------

    labels, MST_connections : ndarry, list
        Lables corresponding to respective classes of points and MST_connections
        defining which pairs of points remain connect in final MST.

    Returns
    -------
    run_count : int
        Number of runs after severing all such edges with nodes of
        differing class labels.

    """

    run_count = 1;

    for x in MST_connections:
        if labels[x[0]] != labels[x[1]]:
            run_count += 1;

    return run_count


@jit(nopython=True, cache=True)
 def permutation(nperm, labels, MST_connections):

    r"""
    Helper function that randomizes labels and calculates respective
    'run' counts for a specified number of iterations.

    """

    runs = []
    for itr in arange(nperm):

        lab_shuffle = random.sample(labels, len(labels))

        run_val = num_runs(lab_shuffle, MST_connections)

        runs.append(run_val)

    return runs

