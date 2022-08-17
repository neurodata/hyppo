from abc import ABC, abstractmethod

import numpy as np


class DiscriminabilityTest(ABC):
    r"""
    A base class for a discriminability test.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def test(self, *args, **kwargs):
        r"""
        Calculates the Discriminability test statistic and p-value.
        """

class PopDiscriminabilityTest(DiscriminabilityTest):
    """
    A base class for a population discriminability test.
    """

    def __init__(self, is_dist=False, remove_isolates=True):
        self.pvalue_ = None
        super.__init__()

    @abstractmethod
    def statistic(self, x, y):
        r"""
        Calulates the independence test statistic.

        Parameters
        ----------
        x, y : ndarray of float
            Input data matrices.
        """
        check_input = _CheckInputs(
            [x],
            y,
            reps=reps,
            is_dist=self.is_distance,
            remove_isolates=self.remove_isolates,
        )
        x, y = check_input()

        self.x = np.asarray(x[0])
        self.y = y

        rdfs = _discr_pop_rdf(x, y)
        stat = np.nanmean(rdfs)

        return stat
    
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

def _discr_pop_rdf(dissimilarities, objects):
    # calculates test statistics distribution
    rdfs = []

    for i, obj in enumerate(objects):
        di = dissimilarities[i]

        # All other samples except its own label
        idx = objects == obj
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

def ClassDiscriminabilityTest(DiscriminabilityTest):
    """
    A base class for a class discriminability test.
    """

    def __init__(self, is_dist=False, remove_isolates=True):
        self.pvalue_ = None
        super.__init__()
        
    @abstractmethod
    def statistic(self, x, y, z):
        r"""
        Calculates the independence test statistic.

        Parameters
        ----------
        x, y, z : ndarray of float
            Input data matrices.
        """
        check_input = _CheckInputs(
            [x],
            y,
            reps=reps,
            is_dist=self.is_distance,
            remove_isolates=self.remove_isolates,
        )
        x, y = check_input()

        self.x = np.asarray(x[0])
        self.y = y
        self.z = z

        weights = []
        zs = np.unique(z)
        K = len(zs)
        N = x.shape[0]

        weights = np.zeros((K, K))
        discrs = np.zeros((K, K))
        for i, z1 in enumerate(zs):
            Nz1 = (z == z1).sum()
            for j, z2 in enumerate(zs):
                if z == zp:
                    Nz2 = Nz1 - 1
                else:
                    Nz2 = (z == z2).sum()
                weights[i, j] = Nz1*Nz2/(N*(N - 1))
                discrs[i, j] = _statistic_zzp(z1=z1, z2=z2)
        return weights, discrs

    def _statistic_zzp(z1, z2):
        r"""
        Calulates the independence test statistic.

        Parameters
        ----------
        x, y : ndarray of float
            Input data matrices.
        """
        rdfs = []
        # isolate analysis to only elements from classes z1 or z2
        idx_z1z2 = np.which(np.logical_or(self.z == z1, self.z == z2))[0]
        y_z1z2 = self.y[idx_z1z2]
        z_z1z1 = self.z[idx_z1z2]
        for i in idx_z1z2:
            # the class label of object i
            z_i = self.z[i]
            # the individual label of object i
            ind_i = self.y[i]
            # get all of the distances from i to other items that have class
            # of z1 or z2, where the individual label is the same
            Dii = self.x[i][idx_z1z2][y_z1z2 == ind_i]
            if z_i == z1:
                z_oth = z2
            else:
                z_oth = z1
            # get all of the distances from i to other items that have
            # class of z1 or z2, where the individual label is different
            # and the class is the class that object i is not    
            Dij = self.x[i][idx_z1z2][z_z1z2 == z_oth]

            rdf = [1 - ((Dij < d).sum() + 0.5 * (Dij == d).sum()) / Dij.size for d in Dii]
            rdfs.append(rdf)
        stat = np.array(rdfs).mean()
        return stat
