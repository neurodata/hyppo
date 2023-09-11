from math import ceil
import numpy as np
from ..independence import INDEP_TESTS
from ..ksample import KSAMP_TESTS, KSample, k_sample_transform
from ..d_variate import MULTI_TESTS
from .indep_sim import indep_sim
from .ksample_sim import gaussian_3samp, rot_ksamp

# Define constants
ALL_SIMS = {
    "indep": indep_sim,
    "multi": indep_sim,
    "ksamp": rot_ksamp,
    "gauss": gaussian_3samp,
}

NONPERM_TESTS = {
    "dcorr": "fast",
    "hsic": "fast",
    "dhsic": "fast",
    "energy": "fast",
    "mmd": "fast",
    "disco": "fast",
    "manova": "analytical",
    "hotelling": "analytical",
    "kmerf": "fast",
}

PERM_STATS = {
    "indep": _indep_perm_stat,
    "ksamp": _ksamp_perm_stat,
    "multi": _multi_perm_stat,
}


def _sim_gen(sim_type, **kwargs):
    """
    Generate ``sims`` for the desired simulations.
    """
    if sim_type in ["indep", "ksamp"]:
        if (
            kwargs["sim"] in ["multiplicative_noise", "multimodal_independence"]
            and "noise" in kwargs.keys()
        ):
            kwargs.pop("noise")

    sims = ALL_SIMS[sim_type](**kwargs)

    return sims


def _indep_perm_stat(test, sim_type, **kwargs):
    """
    Generates null and alternate distributions for the independence test.
    """
    x, y = _sim_gen(sim_type=sim_type, **kwargs)
    obs_stat = test.statistic(x, y)
    permy = np.random.permutation(y)
    perm_stat = test.statistic(x, permy)

    return obs_stat, perm_stat


def _ksamp_perm_stat(test, sim_type, **kwargs):
    """
    Generates null and alternate distributions for the k-sample test.
    """
    sims = _sim_gen(sim_type=sim_type, **kwargs)
    u, v = k_sample_transform(sims)
    obs_stat = test.statistic(u, v)
    permv = np.random.permutation(v)
    perm_stat = test.statistic(u, permv)

    return obs_stat, perm_stat


def _multi_perm_stat(test, sim_type, **kwargs):
    """
    Generates null and alternate distributions for the d_variate independence test.
    """
    x, y = _sim_gen(sim_type=sim_type, **kwargs)
    obs_stat = test.statistic(*(x, y))
    [permx, permy] = np.split(np.random.permutation(np.append(x, y)), 2)
    permx = permx.reshape(permx.shape[0], 1)
    permy = permy.reshape(permy.shape[0], 1)
    perm_stat = test.statistic(*(permx, permy))

    return obs_stat, perm_stat


def _nonperm_pval(test, sim_type, **kwargs):
    """
    Generates fast permutation pvalues
    """
    sims = _sim_gen(sim_type=sim_type, **kwargs)
    pvalue = test.test(*sims)[1]

    return pvalue


def power(test, sim_type, sim=None, n=100, alpha=0.05, reps=1000, auto=False, **kwargs):
    """
    Computes empirical power for hypothesis tests.

    Parameters
    ----------
    test : str or list
        The name of the independence test or k-sample test.
    sim_type : str
        Type of power method to calculate.
    sim : str, default: None
        The name of the independence simulation that is to be used.
    n : int, default: 100
        The number of samples desired by the simulation (>= 5).
    alpha : float, default: 0.05
        The alpha level of the test.
    reps : int, default: 1000
        The number of replications used to estimate the null distribution.
    auto : bool, default: False
        Automatically uses fast approximation when `n` is greater than 20 or test has a non-permutation based p-value.
    **kwargs
        Additional keyword arguments for `sim`.

    Returns
    -------
    empirical_power : ndarray of float
        Estimated empirical power for the test.
    """
    if sim_type not in ALL_SIMS.keys():
        raise ValueError(
            f"sim_type {sim_type}, must be in {list(ALL_SIMS.keys())}"
        )

    # ...

    return empirical_power
