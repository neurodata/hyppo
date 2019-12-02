from sklearn.metrics import euclidean_distances
from ._utils import _CheckInputs
import numpy as np
import random
from .base import discriminabilityTest
from scipy._lib._util import MapWrapper


class twoSample(discriminabilityTest):
    r"""
     A class that performs a two-sample test for whether the discriminability is different for that of
     one dataset vs another, as described in [1]. With :math:`\hat D_{X_1}` the sample discriminability of
     one approach, and :math:`\hat D_{X_2}` the sample discriminability of another approach:

     .. math::
        H_0: D_{X_1} = D_{X_2}

     and

     .. math::
        H_A: D_{X_1} > D_{X_2} 
    
    Also implemented are tests of :math:`<` and :math:`\neq`.
    
    References
    ----------
    .. [#1Dscr_] Eric W. Bridgeford, et al. "Optimal Decisions for Reference 
                Pipelines and Datasets: Applications in Connectomics." Bioarxiv (2019).
    """

    def __init__(self):
        discriminabilityTest.__init__(self)


    def test(self, X1, X2, Y, remove_isolates=True, reps=1000, alt="greater", workers=-1):
        r"""
        Calculates the test statistic and p-value for Discriminability two sample test.

        Parameters
        ----------
        X1 : ndarray

            * An :math:`n \times d` dimensional data matrix with :math:`n` samples in :math:`d` dimensions. Should not be a distance matrix.  

        X2 : ndarray

            * An :math:`n \times d` dimensional data matrix with :math:`n` samples in :math:`d` dimensions. Should not be a distance matrix.

        Y :               ndarray
                          A vector containing the sample ids for our :math:`n` samples. Should be matched such that :math:`Y[i]` 
                          is the corresponding label for :math:`X1[i,]` and :math:`X2[i,]`.
        remove_isolates : Boolean, optional (default: True)
                          whether remove the samples with single instance or not.
        reps :            int, optional (default: 1000)
                          The number of replications used to estimate the null distribution
                          when using the permutation test used to calculate the p-value.
        alt :             string, optional(default: "greater")
                          The alternative hypothesis for the test. Can be that first dataset is more discriminable (alt = "greater"),
                          less discriminable (alt = "less") or just non-equal (alt = "neq").
        workers :         int, optional (default: -1)
                          The number of cores to parallelize the p-value computation over.
                          Supply -1 to use all cores available to the Process.

        Returns
        -------
        D1 :     float
                 The computed discriminability score for :math:`X_1`.
        D2 :     float
                 The computed discriminability score for :math:`X_2`.
        pvalue : float
                 The computed two sample test p-value.
        """
        check_input = _CheckInputs(X1, Y, reps = reps)
        X1, Y = check_input()
        
        _, counts = np.unique(Y, return_counts=True)

        if (counts != 1).sum() <= 1:
            msg = "You have passed a vector containing only a single unique sample id."
            raise ValueError(msg)


        check_input = _CheckInputs(X2, Y, reps = reps)
        X2, Y = check_input()

        _, counts_ = np.unique(Y, return_counts=True)

        if (counts_ != 1).sum() <= 1:
            msg = "You have passed a vector containing only a single unique sample id."
            raise ValueError(msg)
        
        if (counts != 1).sum() != (counts_ != 1).sum():
            msg = "The input matrices do not have the same number of rows."
            raise ValueError(msg)


        self.X1 = X1
        self.X2 = X2
        self.Y = Y
        
        self.D1_ = super(twoSample,self)._statistic(self.X1, self.Y,is_dist = False, remove_isolates = remove_isolates, return_rdfs=False)
        self.D2_ = super(twoSample,self)._statistic(self.X2, self.Y,is_dist = False, remove_isolates = remove_isolates, return_rdfs=False)
        self.Da_ = self.D1_ - self.D2_

        # use all cores to create function that parallelizes over number of reps
        mapwrapper = MapWrapper(workers)
        null_dist = np.array(list(mapwrapper(self._perm_stat, range(reps))))
        self.null_dist = null_dist
        
        self.diffNull = []

        for i in range(0,reps-1):
            for j in range(i+1,reps):
                self.diffNull.append(self.null_dist[i][0] - self.null_dist[j][1])
                self.diffNull.append(self.null_dist[j][1] - self.null_dist[i][0])
        
        self.diffNull = np.asarray(self.diffNull)

        if alt == "greater":
            p = (self.diffNull >= self.Da_).mean()
        elif alt == "less":
            p = (self.diffNull <= self.Da_).mean()
        elif alt == "neq":
            p = (abs(self.diffNull) >= abs(self.Da_).mean())
        else:
            msg = "You have not entered a valid alternative."
            raise ValueError(msg)

        self.pvalue_ = (p*reps + 1)/(1 + reps)

        return self.D1_, self.D2_, self.pvalue_



    def getConvexComb(self, X):
        N, _ = X.shape

        q1 = np.random.choice(N, N)
        q2 = np.random.choice(N, N)
        lamda = np.random.uniform(size=N)

        return (lamda*(X[q1]).T + (1-lamda)*(X[q2]).T).T

    def _perm_stat(self, index):
        permx1 = self.getConvexComb(self.X1)
        permx2 = self.getConvexComb(self.X2)
        permy = self.Y

        perm_stat1 = super(twoSample,self)._statistic(permx1, permy)
        perm_stat2 = super(twoSample,self)._statistic(permx2, permy)

        return perm_stat1, perm_stat2
