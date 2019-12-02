from abc import ABC, abstractmethod
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_X_y
import numpy as np


class discriminabilityTest(ABC):
    r"""
    A base class for a Discriminability test.
    
    """
    
    
    def __init__(self):
        super().__init__()


    def _statistic(self, X, Y, is_dist = False, remove_isolates=True, return_rdfs=False):
        r"""
        Calulates the independence test statistic.
            
        Parameters
        ----------
        x, y : ndarray
        Input data matrices.
        """

        uniques, counts = np.unique(Y, return_counts=True)
        
        if remove_isolates:
            idx = np.isin(Y, uniques[counts != 1])
            labels = Y[idx]
            
            if ~is_dist:
                X = X[idx]
            else:
                X = X[np.ix_(idx, idx)]
        else:
            labels = Y

        if ~is_dist:
            dissimilarities = euclidean_distances(X)
        else:
            dissimilarities = X

        rdfs = self._discr_rdf(dissimilarities, labels)
        stat = np.nanmean(rdfs)

        if return_rdfs:
            return stat, rdfs
        else:
            return stat




    def _discr_rdf(self, dissimilarities, labels):
    
        check_X_y(dissimilarities, labels, accept_sparse=True)

        rdfs = []
        for i, label in enumerate(labels):
            di = dissimilarities[i]

            # All other samples except its own label
            idx = labels == label
            Dij = di[~idx]

             # All samples except itself
            idx[i] = False
            Dii = di[idx]

            rdf = [1 - ((Dij < d).sum() + 0.5 * (Dij == d).sum()) / Dij.size for d in Dii]
            rdfs.append(rdf)

        out = np.full((len(rdfs), max(map(len, rdfs))), np.nan)
        for i, rdf in enumerate(rdfs):
            out[i, : len(rdf)] = rdf

        return out
    


    @abstractmethod
    def _perm_stat(self, index):
        r"""
        Helper function that is used to calculate parallel permuted test
        statistics.
            
        Parameters
        ----------
        index : int
        Iterator used for parallel statistic calculation
            
        Returns
        -------
        perm_stat : float
        Test statistic for each value in the null distribution.
        """


    @abstractmethod
    def test(self, x, y, reps=1000, workers=-1):
        r"""
        Calulates the independence test p-value.
            
        Parameters
        ----------
        x, y : ndarray
        Input data matrices.
        reps : int, optional
        The number of replications used in permutation, by default 1000.
        workers : int, optional
        Evaluates method using `multiprocessing.Pool <multiprocessing>`).
        Supply `-1` to use all cores available to the Process.
            
        Returns
        -------
        stat : float
        The computed independence test statistic.
        pvalue : float
        The pvalue obtained via permutation.
        """
