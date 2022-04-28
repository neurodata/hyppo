import numpy as np
from numba import jit
from hyppo.ksample._utils import _CheckInputs
from hyppo.ksample.base import KSampleTest, KSampleTestOutput
from sklearn.metrics import pairwise_distances
from scipy.stats import ks_2samp

class HHG(KSampleTest):
    r"""
    Fast HHG 2-Sample Test: 
    """
    def __init__(self, compute_distance="euclidean", **kwargs):
       self.compute_distance = compute_distance
       KSampleTest.__init__(self, compute_distance=compute_distance, **kwargs)
       
    def statistic(self, y1, y2):
        y = np.concatenate((y1,y2), axis=0)
        disty = _centerpoint_dist(y, self.compute_distance, 1)
        disty1 = disty[:,0:len(y1)]
        disty2 = disty[:,len(y1):len(y2)+len(y1)]
        pvalues = _distance_score(disty1,disty2)
        stat = min(pvalues)*len(pvalues)
        self.stat = stat
        return stat

    def test(self, x, y):
        """
        x,y : ndarray of float
            Input data matrices. ``y1`` and ``y2`` must have the same number of
            dimensions. That is, the shapes must be ``(n, p)`` and ``(m, p)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions.
        """
        check_input = _CheckInputs(
            inputs=[x, y],
        )
        x, y = check_input()
        
        xy = np.concatenate((x,y), axis=0)
        distxy = _centerpoint_dist(xy, self.compute_distance, 1)
        distx = distxy[:,0:len(x)]
        disty = distxy[:,len(x):len(x)+len(x)]
        pvalues = _distance_score(distx,disty)
        stat = min(pvalues)*len(pvalues)
        return stat
    
def _centerpoint_dist(xy, metric, workers=1, **kwargs):
    disty = pairwise_distances(xy, metric=metric, n_jobs=workers, **kwargs)
    return disty

def _distance_score(distx, disty):
    dist1, dist2 = _extract_distance(distx, disty)
    pvalues = []
    for i in range(len(distx)):
        stat, pvalue = ks_2samp(dist1[i],dist2[i])
        pvalues.append(pvalue)
    return pvalues

@jit(nopython=True)
def _extract_distance(disty1, disty2):
    dist1 = []
    dist2 = []
    for i in range(len(disty1)):
        distancey1 = np.delete(disty1[i],0)
        distancey2 = np.delete(disty2[i],0)
        distancey1 = distancey1.reshape(-1)
        distancey2 = distancey2.reshape(-1)
        dist1.append(distancey1)
        dist2.append(distancey2)
    return dist1, dist2