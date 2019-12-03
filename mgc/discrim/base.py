from abc import ABC, abstractmethod
from sklearn.utils import check_X_y
import numpy as np
from .._utils import euclidean

class DiscriminabilityTest(ABC):
    r"""
    A base class for a Discriminability test.
    
    """
    
    def __init__(self):
        self.pvalue_ = None
        super().__init__()

    def _statistic(self, X, Y, is_dist = False, remove_isolates=True):
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
            
            if not is_dist:
                X = X[idx]
            else:
                X = X[np.ix_(idx, idx)]
        else:
            labels = Y

        if not is_dist:
            dissimilarities = euclidean(X)
        else:
            dissimilarities = X

        rdfs = self._discr_rdf(dissimilarities, labels)
        stat = np.nanmean(rdfs)

        return stat

    def _discr_rdf(self, dissimilarities, labels):
    
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
