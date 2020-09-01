import numpy as np
from math import ceil
from collections import defaultdict
from copy import deepcopy

from scipy._lib._util import check_random_state, MapWrapper
from sklearn.metrics import pairwise_distances

from hyppo.ksample._utils import k_sample_transform
from hyppo.sims import gaussian_4samp_2way


class _ParallelP_4samp_2way(object):
    """
    Helper function to calculate parallel power.
    """

    def __init__(self, test, n, epsilon1=1, epsilon2=1, effect_mask=None, weight=0, case=1, rngs=[],  multiway=False, permute_groups=None, permute_structure='multilevel', sim_kwargs={}, **kwargs):
        self.test = test(**kwargs)

        self.n = n
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.effect_mask = effect_mask
        self.weight = weight
        self.multiway = multiway
        self.rngs = rngs
        self.sim_kwargs = sim_kwargs

        # New
#         self.permute_groups = np.vstack(([[0,0]]*n, [[0,1]]*n, [[1,1]]*n, [[1,0]]*n))
#         self.permute_structure = permute_structure


    def __call__(self, index):
        Xs = gaussian_4samp_2way(
            self.n,
            epsilon1=self.epsilon1,
            epsilon2=self.epsilon2,
            effect_mask=self.effect_mask,
            **self.sim_kwargs
        )

        if self.multiway:
            ways = [[0,0], [0,1], [1,1], [1,0]]
            u, v = k_sample_transform(Xs, ways=ways)
        else:
            u, v = k_sample_transform(Xs)

        u_dist = pairwise_distances(u, metric="euclidean")
        v_dist = pairwise_distances(v, metric="sqeuclidean")

        obs_stat = self.test._statistic(u_dist, v_dist)

#        self.v_labels = np.unique(self.permute_groups[:,0], return_inverse=True)[1]
#         if self.permute_structure == 'multilevel':
#             # permute_groups [highest level,...,lowest level]
#             # TODO more than 2level, should be generically generalizable
#             self.within_indices = defaultdict(list)
#             for i,group in enumerate(self.permute_groups):
#                 self.within_indices[group[1]].append(i) # lowest level

#             # dict: [y_label] -> list(indices)
#             self.class_indices = defaultdict(list) 
#             across_indices = defaultdict(list)
#             for i,(group,label) in enumerate(zip(self.permute_groups, self.v_labels)):
#                 self.class_indices[label].append(i)
#                 across_indices[group[0]].append(i) # highest level
#             # list of group indices, sorted descending order
#             self.across_indices = sorted(
#                 across_indices.values(), key=lambda x: len(x), reverse=True
#             )
#         else:
#             msg = "permute_structure must be of {'multilevel'}"
#             raise ValueError(msg)
        
        permv = self.rngs[index].permutation(np.arange(len(v)))
        permv = self.rngs[index].permutation(v)

        # calculate permuted stats, store in null distribution
        permv_dist = pairwise_distances(permv, metric="sqeuclidean")
        perm_stat = self.test._statistic(u_dist, permv_dist)

        return obs_stat, perm_stat


def _perm_test_4samp_2way(
    test,
    n=100,
    epsilon1=1,
    epsilon2=1,
    effect_mask=None,
    weight=0,
    case=1,
    reps=1000,
    workers=1,
    random_state=None,
    multiway=False,
    sim_kwargs={},
    **kwargs,
):
    r"""
    Helper function that calculates the statistical.

    Parameters
    ----------
    test : callable()
        The independence test class requested.
    sim : callable()
        The simulation used to generate the input data.
    reps : int, optional (default: 1000)
        The number of replications used to estimate the null distribution
        when using the permutation test used to calculate the p-value.
    workers : int, optional (default: -1)
        The number of cores to parallelize the p-value computation over.
        Supply -1 to use all cores available to the Process.

    Returns
    -------
    null_dist : list
        The approximated null distribution.
    """
    # set seeds
    random_state = check_random_state(random_state)
    rngs = [
        np.random.RandomState(random_state.randint(1 << 32, size=4, dtype=np.uint32))
        for _ in range(reps)
    ]

    # use all cores to create function that parallelizes over number of reps
    mapwrapper = MapWrapper(workers)
    parallelp = _ParallelP_4samp_2way(
        test,
        n,
        epsilon1,
        epsilon2,
        effect_mask,
        weight,
        case,
        rngs,
        multiway,
        sim_kwargs,
        **kwargs
    )
    alt_dist, null_dist = map(list, zip(*list(mapwrapper(parallelp, range(reps)))))
    alt_dist = np.array(alt_dist)
    null_dist = np.array(null_dist)

    return alt_dist, null_dist


def power_4samp_2way_epsweight(
    test,
    n=100,
    epsilon1=0.5,
    epsilon2=0.5,
    effect_mask=None,
    weight=0,
    case=1,
    alpha=0.05,
    reps=1000,
    workers=1,
    random_state=None,
    multiway=False,
    sim_kwargs={},
    **kwargs,
):
    alt_dist, null_dist = _perm_test_4samp_2way(
        test,
        n=n,
        epsilon1=epsilon1,
        epsilon2=epsilon2,
        effect_mask=effect_mask,
        weight=weight,
        case=case,
        reps=reps,
        workers=workers,
        random_state=random_state,
        multiway=multiway,
        sim_kwargs=sim_kwargs,
        **kwargs,
    )
    cutoff = np.sort(null_dist)[ceil(reps * (1 - alpha))]
    empirical_power = (alt_dist >= cutoff).sum() / reps

    if empirical_power == 0:
        empirical_power = 1 / reps

    return empirical_power
