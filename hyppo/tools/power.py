from math import ceil

import numpy as np

from ..independence import INDEP_TESTS
from ..ksample import KSAMP_TESTS, KSample, k_sample_transform
from .indep_sim import indep_sim
from .ksample_sim import gaussian_3samp, rot_ksamp

_ALL_SIMS = {
    "indep": indep_sim,
    "ksamp": rot_ksamp,
    "gauss": gaussian_3samp,
}

_NONPERM_TESTS = {
    "dcorr": "fast",
    "hsic": "fast",
    "energy": "fast",
    "mmd": "fast",
    "disco": "fast",
    "manova": "analytical",
    "hotelling": "analytical",
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
            kwargs = kwargs.copy().pop("noise")

    sims = _ALL_SIMS[sim_type](**kwargs)

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


_PERM_STATS = {
    "indep": _indep_perm_stat,
    "ksamp": _ksamp_perm_stat,
}


def _nonperm_pval(test, sim_type, **kwargs):
    """
    Generates fast  permutation pvalues
    """
    sims = _sim_gen(sim_type=sim_type, **kwargs)
    pvalue = test.test(*sims)[1]

    return pvalue


def power(test, sim_type, sim=None, n=100, alpha=0.05, reps=1000, auto=False, **kwargs):
    """
    Computes empircal power for k-sample tests

    Parameters
    ----------
    test : str or list
        The name of the independence test (from the :mod:`hyppo.independence` module)
        that is to be tested. If MaxMargin, accepts list with first entry "MaxMargin"
        and second entry the name of another independence test.
        For :class:`hyppo.ksample.KSample` put the name of the independence test.
        For other tests in :mod:`hyppo.ksample` just use the name of the class.
    sim_type : "indep", "ksamp", "gauss"
        Type of power method to calculate. Depends on the type of ``sim``.
    sim : str, default: None
        The name of the independence simulation (from the :mod:`hyppo.tools` module).
        that is to be used. Set to ``None`` if using gaussian simulation curve.
    n : int, default: 100
        The number of samples desired by the simulation (>= 5).
    alpha : float, default: 0.05
        The alpha level of the test.
    reps : int, default: 1000
        The number of replications used to estimate the null distribution
        when using the permutation test used to calculate the p-value.
    auto : bool, default: False
        Automatically uses fast approximation when `n` and size of array
        is greater than 20 or test has a non-permutation based p-value.
        If ``True``, and sample size is greater than 20, then
        :class:`hyppo.tools.chi2_approx` will be run. ``reps`` is
        irrelevant in this case.
        See documentation for ``test`` if this parameter applies.
    **kwargs
        Additional keyword arguements for ``sim``.

    Returns
    -------
    empirical_power : ndarray
        Estimated empirical power for the test.
    """
    if sim_type not in _ALL_SIMS.keys():
        raise ValueError(
            "sim_type {}, must be in {}".format(sim_type, list(_ALL_SIMS.keys()))
        )

    if type(test) is list:
        test_name = [t.lower() for t in test]
        if test_name[0] != "maxmargin" or test_name[1] not in INDEP_TESTS.keys():
            raise ValueError(
                "Test 1 must be Maximal Margin, currently {}; Test 2 must be an "
                "independence test, currently {}".format(*test)
            )
        test = INDEP_TESTS[test_name[0]](indep_test=test_name[1])
        test_name = test_name[1]
    else:
        test_name = test.lower()
        if test_name in INDEP_TESTS.keys():
            test = INDEP_TESTS[test_name]()
        elif test_name in KSAMP_TESTS.keys():
            test = KSAMP_TESTS[test_name]()
        else:
            raise ValueError(
                "Test {} not in {}".format(
                    test_name, list(INDEP_TESTS.keys()) + list(KSAMP_TESTS.keys())
                )
            )

    kwargs["n"] = n
    perm_type = "indep"
    if sim_type in ["ksamp", "gauss"]:
        perm_type = "ksamp"
    if sim_type != "gauss":
        kwargs["sim"] = sim

    if test_name in _NONPERM_TESTS.keys() and (
        auto or _NONPERM_TESTS[test_name] == "analytical"
    ):
        if test_name in INDEP_TESTS.keys() and perm_type == "ksamp":
            test = KSample(test_name)
        if n < 20 and _NONPERM_TESTS[test_name] == "fast":
            empirical_power = np.nan
        else:
            pvals = np.array(
                [
                    _nonperm_pval(test=test, sim_type=sim_type, **kwargs)
                    for _ in range(reps)
                ]
            )
            empirical_power = (1 + (pvals <= alpha).sum()) / (1 + reps)
    else:
        alt_dist, null_dist = map(
            np.float64,
            zip(
                *[
                    _PERM_STATS[perm_type](test=test, sim_type=sim_type, **kwargs)
                    for _ in range(reps)
                ]
            ),
        )
        cutoff = np.sort(null_dist)[ceil(reps * (1 - alpha))]
        empirical_power = (1 + (alt_dist >= cutoff).sum()) / (1 + reps)

    return empirical_power
