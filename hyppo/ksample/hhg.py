import numpy as np
from numba import jit
from hyppo.ksample._utils import _CheckInputs
from hyppo.ksample.base import KSampleTest, KSampleTestOutput
from sklearn.metrics import pairwise_distances
from scipy.stats import ks_2samp

class HHG(KSampleTest):
    r"""
    Fast HHG 2-Sample Test
    """
    def __init__(self, compute_distance="euclidean", mode ="CM", **kwargs):
       self.compute_distance = compute_distance
       self.mode = "CM"
       KSampleTest.__init__(self, compute_distance=compute_distance, **kwargs)
       
    def statistic(self, y1, y2):
        if self.mode == "CM":
            disty1, disty2 = _centerpoint_dist_CM(y1, y2, self.compute_distance, 1)
            stat, pvalue = ks_2samp(disty1, disty2)
            self.stat = stat
            return stat
        if self.mode == "MP":
            y = np.concatenate((y1,y2), axis=0)
            disty = _centerpoint_dist_MP(y, self.compute_distance, 1)
            disty1 = disty[:,0:len(y1)]
            disty2 = disty[:,len(y1):len(y2)+len(y1)]
            pvalues = _distance_score_MP(disty1,disty2)
            stat = min(pvalues)*len(pvalues)
            self.stat = stat
            return stat

    def test(self, y1, y2):
        """
        y1,y2 : ndarray of float
            Input data matrices. ``y1`` and ``y2`` must have the same number of
            dimensions. That is, the shapes must be ``(n, p)`` and ``(m, p)`` where
            `n` is the number of samples and `p` and `q` are the number of
            dimensions.
        """
        if self.mode == "CM":
            disty1, disty2 = _centerpoint_dist_CM(y1, y2, self.compute_distance, 1)
            stat, pvalue = ks_2samp(disty1,disty2)
            return stat, pvalue
        elif self.mode == "MP":
            y = np.concatenate((y1,y2), axis=0)
            disty = _centerpoint_dist_MP(y, self.compute_distance, 1)
            disty1 = disty[:,0:len(y1)]
            disty2 = disty[:,len(y1):len(y2)+len(y1)]
            pvalues = _distance_score_MP(disty1,disty2)
            stat = min(pvalues)*len(pvalues)
            return stat
        
def _centerpoint_dist_CM(y1, y2, metric, workers=1, **kwargs):
    #CM of one group as center point
    zy = np.mean(y1, axis=0)
    zy = np.array(zy).reshape(1,-1)
    yin1 = np.concatenate((zy, y1))
    yin2 = np.concatenate((zy, y2))
    if callable(metric):
        disty1 = metric(yin1, **kwargs)
        disty1 = metric(yin2, **kwargs)
    else:
        disty1 = pairwise_distances(yin1, metric=metric, n_jobs=workers, **kwargs)
        disty2 = pairwise_distances(yin2, metric=metric, n_jobs=workers, **kwargs)
    disty1 = np.delete(disty1[0],0)
    disty2 = np.delete(disty2[0],0)
    disty1 = disty1.reshape(-1)
    disty2 = disty2.reshape(-1)
    return disty1, disty2
    
def _centerpoint_dist_MP(y, metric, workers=1, **kwargs):
    disty = pairwise_distances(y, metric=metric, n_jobs=workers, **kwargs)
    return disty

def _distance_score_MP(disty1, disty2):
    dist1, dist2 = _extract_distance_MP(disty1, disty2)
    pvalues = []
    for i in range(len(disty1)):
        stat, pvalue = ks_2samp(dist1[i],dist2[i])
        pvalues.append(pvalue)
    return pvalues

@jit(nopython=True)
def _extract_distance_MP(disty1, disty2):
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