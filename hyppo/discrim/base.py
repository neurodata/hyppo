from abc import ABC, abstractmethod

import numpy as np


class DiscriminabilityTest(ABC):
    r"""
    A base class for a Discriminability test.
    """

    def __init__(self):
        self.pvalue_ = None
        super().__init__()

    @abstractmethod
    def _statistic(self, x, y):
        r"""
        Calulates the independence test statistic.

        Parameters
        ----------
        x, y : ndarray
            Input data matrices.
        """

        rdfs = self._discr_rdf(x, y)
        stat = np.nanmean(rdfs)

        return stat

    def _discr_rdf(self, dissimilarities, labels):
        # calculates test statistics distribution
        rdfs = []

        for i, label in enumerate(labels):
            di = dissimilarities[i]

            # All other samples except its own label
            idx = labels == label
            Dij = di[~idx]

            # All samples except itself
            idx[i] = False
            Dii = di[idx]

            rdf = [
                1 - ((Dij < d).sum() + 0.5 * (Dij == d).sum()) / Dij.size for d in Dii
            ]
            rdfs.append(rdf)

        out = np.full((len(rdfs), max(map(len, rdfs))), np.nan)
        for i, rdf in enumerate(rdfs):
            out[i, : len(rdf)] = rdf

        return out

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
    def test(self):
        r"""
        Calculates the test statistic and p-value for Discriminability one sample
        and two sample test.
        """
