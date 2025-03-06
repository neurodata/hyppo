import numpy as np
from sklearn.utils import check_random_state
from scipy import stats

from .indep_sim import _CheckInputs


def simulate_covars_binary(groups, balance=1, random_state=None):
    """
    Simulate Binary Covariates

    Parameters
    ----------
    groups : array-like
        Group assignments as an [n] vector, which take values from 0 to K for K groups.
    balance : float, default: 1
        A parameter that governs the similarity between the binary covariate distributions
        for the 0 group against the other K-1 groups.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    covars : ndarray
        An [n] vector containing binary predictors where the 0 group is sampled from
        Bern(balance/2) and the K-1 groups are sampled from Bern(1 - balance/2).
    """
    n = len(groups)
    rng = check_random_state(random_state)

    covars = np.zeros(n)
    plow = balance / 2
    phigh = 1 - plow

    for i in range(n):
        if groups[i] == 0:
            covars[i] = rng.binomial(1, plow)
        else:
            covars[i] = rng.binomial(1, phigh)

    return covars


def simulate_covars(groups, balance=1, alpha=2, beta=8, common=10, random_state=None):
    """
    Simulate Continuous Covariates

    Parameters
    ----------
    groups : array-like
        Group assignments as an [n] vector, which take values from 0 to K for K groups.
    balance : float, default: 1
        A parameter that governs the similarity between the continuous covariate distributions
        for the 0 group against the other K-1 groups.
    alpha : float, default: 2
        The alpha for sampling the 0 group, and the beta for sampling the other K-1 groups.
    beta : float, default: 8
        The beta for sampling the 0 group, and the alpha for sampling the other K-1 groups.
    common : float, default: 10
        A parameter which governs the shape of the common sampling distribution.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    covars : ndarray
        An [n] vector containing continuous predictors where the 0 group is sampled from
        Beta(alpha, beta) with probability 1-balance and the K-1 groups are sampled from
        Beta(beta, alpha) with probability 1-balance, and both groups are sampled from
        Beta(common, common) with probability balance.
    """
    n = len(groups)
    rng = check_random_state(random_state)

    balance_id = rng.binomial(1, balance, n)
    coefs = []

    for i in range(n):
        causal_cl = groups[i]
        bal_id = balance_id[i]

        if bal_id == 1:
            coefs.append((common, common))
        else:
            if causal_cl == 0:
                coefs.append((alpha, beta))
            else:
                coefs.append((beta, alpha))

    covars = np.zeros(n)
    for i in range(n):
        alpha_val, beta_val = coefs[i]
        covars[i] = 2 * rng.beta(alpha_val, beta_val) - 1

    return covars


def simulate_covars_multiclass(
    groups, balance=1, alpha=2, beta=8, common=10, random_state=None
):
    """
    Simulate Continuous Covariates for Multiple Classes

    Parameters
    ----------
    groups : array-like
        Group assignments as an [n] vector, which take values from 0 to K-1 for K groups.
    balance : float, default: 1
        A parameter that governs the similarity between the continuous covariate distributions
        for the 0 group against the other K-1 groups.
    alpha : float, default: 2
        The alpha for sampling the 0 group, and the beta for sampling the other K-1 groups.
    beta : float, default: 8
        The beta for sampling the 0 group, and the alpha for sampling the other K-1 groups.
    common : float, default: 10
        A parameter which governs the shape of the common sampling distribution.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    covars : ndarray
        An [n] vector containing continuous predictors where the 0 group is sampled from
        Beta(alpha, beta) with probability 1-balance and the K-1 groups are sampled from
        Beta(beta, alpha) with probability 1-balance, and all groups are sampled from
        Beta(common, common) with probability balance.
    """
    n = len(groups)
    rng = check_random_state(random_state)

    balance_id = rng.binomial(1, balance, n)
    coefs = []

    for i in range(n):
        causal_cl = groups[i]
        bal_id = balance_id[i]

        if bal_id == 1:
            coefs.append((common, common))
        else:
            if causal_cl == 0:
                coefs.append((alpha, beta))
            else:
                coefs.append((beta, alpha))

    covars = np.zeros(n)
    for i in range(n):
        alpha_val, beta_val = coefs[i]
        covars[i] = 2 * rng.beta(alpha_val, beta_val) - 1

    return covars


def random_rotation(p, random_state=None):
    """
    Generate a Random Orthogonal Matrix

    Creates a random orthogonal matrix from the special orthogonal group SO(n),
    which is the group of n×n orthogonal matrices with determinant 1.
    This is equivalent to scipy's `special_ortho_group.rvs` function.

    Parameters
    ----------
    p : int
        Integer specifying the dimension of the square matrix to generate.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    Q : ndarray
        A p×p orthogonal matrix with determinant 1.

    Notes
    -----
    The function generates a random matrix with standard normal entries,
    performs QR decomposition to obtain an orthogonal matrix,
    and ensures the determinant is 1 by potentially flipping a column sign.
    """
    # Check input
    if not isinstance(p, int) or p < 1:
        raise ValueError("Dimension p must be an integer >= 1")

    rng = check_random_state(random_state)

    # Generate a random matrix with standard normal entries
    M = rng.normal(size=(p, p))

    # QR decomposition
    Q, R = np.linalg.qr(M)

    # Ensure the determinant is 1 (special orthogonal group)
    if np.linalg.det(Q) < 0:
        # Flip the sign of the first column if determinant is negative
        Q[:, 0] = -Q[:, 0]

    return Q


def sigmoid(x):
    """
    Sigmoid function

    Parameters
    ----------
    x : array-like
        Input values

    Returns
    -------
    y : ndarray
        Sigmoid of input values
    """
    return 1 / (1 + np.exp(-x))


def sigmoidal_sim(
    n=100,
    p=10,
    pi=0.5,
    balance=1,
    eff_sz=1,
    covar_eff_sz=5,
    alpha=2,
    beta=8,
    common=10,
    a=2,
    b=8,
    err=1,
    nbreaks=200,
    rotate=True,
    random_state=None,
):
    """
    Sigmoidal CATE Simulation

    Parameters
    ----------
    n : int, default: 100
        The number of samples.
    p : int, default: 10
        The number of dimensions.
    pi : float, default: 0.5
        The balance between the classes, where samples will be from group 1
        with probability pi, and group 0 with probability 1-pi.
    balance : float, default: 1
        A parameter governing the covariate similarity between the two groups.
        Value of 1 means the same covariate distributions for both groups.
    eff_sz : float, default: 1
        The conditional treatment effect size between the different groups, which
        governs the rotation in radians between the first and second group.
    covar_eff_sz : float, default: 5
        A parameter which governs the covariate effect size with respect to the outcome.
    alpha : float, default: 2
        The alpha for sampling the 0 group, and the beta for sampling the other K-1 groups.
    beta : float, default: 8
        The beta for sampling the 0 group, and the alpha for sampling the other K-1 groups.
    common : float, default: 10
        A parameter which governs the shape of the common sampling distribution.
    a : float, default: 2
        The first parameter for the covariate/outcome relationship.
    b : float, default: 8
        The second parameter for the covariate/outcome relationship.
    err : float, default: 1
        The level of noise for the simulation.
    nbreaks : int, default: 200
        The number of breakpoints for computing the expected outcome at a given covariate level
        for each batch.
    rotate : bool, default: True
        Whether to apply a random rotation to the outcomes.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - Ys : ndarray, shape (n, p)
            The outcomes for each sample.
        - Ts : ndarray, shape (n,)
            The group labels for each sample.
        - Xs : ndarray, shape (n, 1)
            The covariate values for each sample.
        - Eps : ndarray, shape (n, p)
            The error for each sample.
        - Ytrue : ndarray
            The expected outcomes at a covariate level indicated by Xtrue.
        - Ttrue : ndarray
            The group/batch the expected outcomes and covariate breakpoints correspond to.
        - Xtrue : ndarray
            The values of the covariate breakpoints for the theoretical expected outcome in Ytrue.
        - Group.Effect : float
            The group effect magnitude.
        - Covar.Effect : float
            The covariate effect magnitude.
        - R : ndarray, optional
            The rotation matrix applied if rotate=True.

    References
    ----------
    Eric W. Bridgeford, et al. "Learning Sources of Variability from High-Dimensional
    Observational Studies" arXiv (2025).
    """
    # Check input parameters
    check_in = _CheckInputs(n, p)
    check_in()

    rng = check_random_state(random_state)

    rotation = eff_sz * np.pi  # Angle of rotation of the second group
    rot_rescale = np.cos(
        rotation
    )  # The rescaling factor for the rotation of the second group

    Ts = rng.binomial(1, pi, n)
    Xs = simulate_covars(
        Ts, balance=balance, alpha=alpha, beta=beta, common=common, random_state=rng
    )
    y_base = sigmoid(b * Xs).reshape(-1, 1)
    Bs = a / (np.arange(1, p + 1) ** 1.5)
    Bs = Bs.reshape(1, -1)

    Ys_covar = covar_eff_sz * (y_base @ Bs)

    rot_vec = np.ones(n)
    rot_vec[Ts == 0] = rot_rescale
    R_diag = np.diag(rot_vec)

    Ys_covar = R_diag @ (Ys_covar - covar_eff_sz / 2 * np.tile(Bs, (n, 1)))
    Ys_covar = Ys_covar + covar_eff_sz / 2 * np.tile(Bs, (n, 1))
    eps = rng.normal(0, err, size=(n, p))
    Ys = Ys_covar + eps

    # True signal at a given x
    true_x = np.linspace(-1, 1, nbreaks)
    true_x = np.concatenate([true_x, true_x])
    true_y_base = sigmoid(b * true_x).reshape(-1, 1)
    true_y_covar = covar_eff_sz * (true_y_base @ Bs)
    true_t = np.concatenate([np.zeros(nbreaks), np.ones(nbreaks)])
    rot_vec_true = np.ones(2 * nbreaks)
    rot_vec_true[true_t == 0] = rot_rescale
    R_true = np.diag(rot_vec_true)
    true_y = R_true @ (true_y_covar - covar_eff_sz / 2 * np.tile(Bs, (2 * nbreaks, 1)))
    true_y = true_y + covar_eff_sz / 2 * np.tile(Bs, (2 * nbreaks, 1))

    out = {
        "Ys": Ys,
        "Ts": Ts,
        "Xs": Xs.reshape(-1, 1),
        "Eps": eps,
        "Ytrue": true_y,
        "Ttrue": true_t,
        "Xtrue": true_x,
        "Group.Effect": eff_sz,
        "Covar.Effect": covar_eff_sz,
    }

    if rotate:
        # If desired, generate and apply a rotation matrix
        R = random_rotation(p, random_state=rng)
        out["Ys"] = out["Ys"] @ R.T
        out["R"] = R

    return out


def nonmonotone_sim(
    n=100,
    p=10,
    pi=0.5,
    balance=1,
    eff_sz=1,
    alpha=2,
    beta=8,
    common=10,
    err=1,
    nbreaks=200,
    rotate=True,
    random_state=None,
):
    """
    Non-monotone CATE Simulation

    Parameters
    ----------
    n : int, default: 100
        The number of samples.
    p : int, default: 10
        The number of dimensions.
    pi : float, default: 0.5
        The balance between the classes, where samples will be from group 1
        with probability pi, and group 0 with probability 1-pi.
    balance : float, default: 1
        A parameter governing the covariate similarity between the two groups.
        Value of 1 means the same covariate distributions for both groups.
    eff_sz : float, default: 1
        The conditional treatment effect size between the different groups.
    alpha : float, default: 2
        The alpha for sampling the 0 group, and the beta for sampling the other K-1 groups.
    beta : float, default: 8
        The beta for sampling the 0 group, and the alpha for sampling the other K-1 groups.
    common : float, default: 10
        A parameter which governs the shape of the common sampling distribution.
    err : float, default: 1
        The level of noise for the simulation.
    nbreaks : int, default: 200
        The number of breakpoints for computing the expected outcome at a given covariate level
        for each batch.
    rotate : bool, default: True
        Whether to apply a random rotation to the outcomes.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - Ys : ndarray, shape (n, p)
            The outcomes for each sample.
        - Ts : ndarray, shape (n,)
            The group labels for each sample.
        - Xs : ndarray, shape (n, 1)
            The covariate values for each sample.
        - Eps : ndarray, shape (n, p)
            The error for each sample.
        - Ytrue : ndarray
            The expected outcomes at a covariate level indicated by Xtrue.
        - Ttrue : ndarray
            The group/batch the expected outcomes and covariate breakpoints correspond to.
        - Xtrue : ndarray
            The values of the covariate breakpoints for the theoretical expected outcome in Ytrue.
        - Group.Effect : float
            The group effect magnitude.
        - R : ndarray, optional
            The rotation matrix applied if rotate=True.

    References
    ----------
    Eric W. Bridgeford, et al. "Learning Sources of Variability from High-Dimensional
    Observational Studies" arXiv (2025).
    """
    # Check input parameters
    check_in = _CheckInputs(n, p)
    check_in()

    rng = check_random_state(random_state)

    Ts = rng.binomial(1, pi, n)
    Xs = simulate_covars(
        Ts, balance=balance, alpha=alpha, beta=beta, common=common, random_state=rng
    )

    y_base = Xs.reshape(-1, 1)
    Bs = 2 / (np.arange(1, p + 1) ** 1.5)
    Bs = Bs.reshape(1, -1)

    Ys_covar = np.zeros((n, p))
    idx = np.where((Xs >= -0.3) & (Xs <= 0.3))[0]
    Ys_covar[idx, :] = eff_sz * np.tile(Bs, (len(idx), 1))
    idx_0 = np.where(Ts == 0)[0]
    Ys_covar[idx_0, :] = -Ys_covar[idx_0, :]

    eps = rng.normal(0, err, size=(n, p))
    Ys = Ys_covar + eps

    # True signal at a given x
    true_x = np.linspace(-1, 1, nbreaks)
    true_x = np.concatenate([true_x, true_x])
    true_y_covar = np.zeros((2 * nbreaks, p))
    idx = np.where((true_x >= -0.3) & (true_x <= 0.3))[0]
    true_y_covar[idx, :] = eff_sz * np.tile(Bs, (len(idx), 1))
    true_t = np.concatenate([np.zeros(nbreaks), np.ones(nbreaks)])
    idx_0 = np.where(true_t == 0)[0]
    true_y_covar[idx_0, :] = -true_y_covar[idx_0, :]

    true_y = true_y_covar

    out = {
        "Ys": Ys,
        "Ts": Ts,
        "Xs": Xs.reshape(-1, 1),
        "Eps": eps,
        "Ytrue": true_y,
        "Ttrue": true_t,
        "Xtrue": true_x,
        "Group.Effect": eff_sz,
    }

    if rotate:
        # If desired, generate and apply a rotation matrix
        R = random_rotation(p, random_state=rng)
        out["Ys"] = out["Ys"] @ R.T
        out["R"] = R

    return out


def kclass_sigmoidal_sim(
    n=100,
    p=10,
    pi=0.5,
    balance=1,
    eff_sz=1,
    covar_eff_sz=5,
    alpha=2,
    beta=8,
    common=10,
    a=2,
    b=8,
    err=1,
    nbreaks=200,
    K=3,
    rotate=True,
    random_state=None,
):
    """
    K-class Sigmoidal CATE Simulation

    Parameters
    ----------
    n : int, default: 100
        The number of samples.
    p : int, default: 10
        The number of dimensions.
    pi : float, default: 0.5
        The balance between the classes, where samples will be from group 0
        with probability pi, and groups 1 to K-1 with equal probability (1-pi)/(K-1).
    balance : float, default: 1
        A parameter governing the covariate similarity between the groups.
        Value of 1 means the same covariate distributions for all groups.
    eff_sz : float, default: 1
        The conditional treatment effect size between the different groups, which
        governs the rotation in radians between the first and other groups.
    covar_eff_sz : float, default: 5
        A parameter which governs the covariate effect size with respect to the outcome.
    alpha : float, default: 2
        The alpha for sampling the 0 group, and the beta for sampling the other K-1 groups.
    beta : float, default: 8
        The beta for sampling the 0 group, and the alpha for sampling the other K-1 groups.
    common : float, default: 10
        A parameter which governs the shape of the common sampling distribution.
    a : float, default: 2
        The first parameter for the covariate/outcome relationship.
    b : float, default: 8
        The second parameter for the covariate/outcome relationship.
    err : float, default: 1
        The level of noise for the simulation.
    nbreaks : int, default: 200
        The number of breakpoints for computing the expected outcome at a given covariate level
        for each batch.
    K : int, default: 3
        The number of classes.
    rotate : bool, default: True
        Whether to apply a random rotation to the outcomes.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - Ys : ndarray, shape (n, p)
            The outcomes for each sample.
        - Ts : ndarray, shape (n,)
            The group labels for each sample.
        - Xs : ndarray, shape (n, 1)
            The covariate values for each sample.
        - Eps : ndarray, shape (n, p)
            The error for each sample.
        - Ytrue : ndarray
            The expected outcomes at a covariate level indicated by Xtrue.
        - Ttrue : ndarray
            The group/batch the expected outcomes and covariate breakpoints correspond to.
        - Xtrue : ndarray
            The values of the covariate breakpoints for the theoretical expected outcome in Ytrue.
        - Group.Effect : float
            The group effect magnitude.
        - Covar.Effect : float
            The covariate effect magnitude.
        - K : int
            The total number of classes.
        - R : ndarray, optional
            The rotation matrix applied if rotate=True.

    References
    ----------
    Eric W. Bridgeford, et al. "Learning Sources of Variability from High-Dimensional
    Observational Studies" arXiv (2025).
    """
    # Check input parameters
    check_in = _CheckInputs(n, p)
    check_in()

    rng = check_random_state(random_state)

    rotation = eff_sz * np.pi  # Angle of rotation of the second group
    rot_rescale = np.cos(
        rotation
    )  # The rescaling factor for the rotation of the second group

    probs = np.array([pi] + [(1 - pi) / (K - 1)] * (K - 1))
    Ts = rng.choice(np.arange(K), size=n, p=probs)
    Xs = simulate_covars_multiclass(
        Ts, balance=balance, alpha=alpha, beta=beta, common=common, random_state=rng
    )
    y_base = sigmoid(b * Xs).reshape(-1, 1)
    Bs = a / (np.arange(1, p + 1) ** 1.1)
    Bs = Bs.reshape(1, -1)

    Ys_covar = covar_eff_sz * (y_base @ Bs)

    rot_vec = np.ones(n)
    rot_vec[Ts == 0] = rot_rescale
    R_diag = np.diag(rot_vec)

    Ys_covar = R_diag @ (Ys_covar - covar_eff_sz / 2 * np.tile(Bs, (n, 1)))
    Ys_covar = Ys_covar + covar_eff_sz / 2 * np.tile(Bs, (n, 1))
    eps = rng.normal(0, err, size=(n, p))
    Ys = Ys_covar + eps

    # True signal at a given x
    Ntrue = K * nbreaks
    true_x = np.linspace(-1, 1, nbreaks)
    true_x = np.tile(true_x, K)
    true_y_base = sigmoid(b * true_x).reshape(-1, 1)
    true_y_covar = covar_eff_sz * (true_y_base @ Bs)
    true_t = np.concatenate([np.full(nbreaks, k) for k in range(K)])
    rot_vec_true = np.ones(Ntrue)
    rot_vec_true[true_t == 0] = rot_rescale
    R_true = np.diag(rot_vec_true)
    true_y = R_true @ (true_y_covar - covar_eff_sz / 2 * np.tile(Bs, (Ntrue, 1)))
    true_y = true_y + covar_eff_sz / 2 * np.tile(Bs, (Ntrue, 1))

    out = {
        "Ys": Ys,
        "Ts": Ts,
        "Xs": Xs.reshape(-1, 1),
        "Eps": eps,
        "Ytrue": true_y,
        "Ttrue": true_t,
        "Xtrue": true_x,
        "Group.Effect": eff_sz,
        "Covar.Effect": covar_eff_sz,
        "K": K,
    }

    if rotate:
        # If desired, generate and apply a rotation matrix
        R = random_rotation(p, random_state=rng)
        out["Ys"] = out["Ys"] @ R.T
        out["R"] = R

    return out


def heteroskedastic_sigmoidal_sim(
    n=100,
    p=10,
    pi=0.5,
    balance=1,
    eff_sz=1,
    covar_eff_sz=3,
    alpha=2,
    beta=8,
    common=10,
    a=2,
    b=8,
    err=0.5,
    nbreaks=200,
    rotate=True,
    random_state=None,
):
    """
    Simulate Data with Heteroskedastic Conditional Average Treatment Effects

    Parameters
    ----------
    n : int, default: 100
        Number of samples to generate.
    p : int, default: 10
        Number of dimensions for the response variable.
    pi : float, default: 0.5
        Probability of assignment to treatment group 1.
    balance : float, default: 1
        A parameter that governs the similarity between the covariate distributions.
    eff_sz : float, default: 1
        Effect size parameter controlling the heteroskedasticity between groups.
    covar_eff_sz : float, default: 3
        Effect size parameter for the covariate influence.
    alpha : float, default: 2
        The alpha parameter for beta distribution of control group.
    beta : float, default: 8
        The beta parameter for beta distribution of control group.
    common : float, default: 10
        Parameter governing the shape of the common sampling distribution.
    a : float, default: 2
        Scaling factor for response variable generation.
    b : float, default: 8
        Scaling factor for sigmoid transformation.
    err : float, default: 0.5
        Standard deviation for the error term.
    nbreaks : int, default: 200
        Number of points to use for generating the true signal.
    rotate : bool, default: True
        Whether to apply a random rotation to the outcomes.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    dict
        A dictionary containing:
        - Ys : ndarray, shape (n, p)
            Response matrix.
        - Ts : ndarray, shape (n,)
            Treatment assignment vector.
        - Xs : ndarray, shape (n, 1)
            Covariate vector.
        - Eps : ndarray, shape (n, p)
            Error matrix.
        - Ytrue : ndarray
            True response matrix for evaluation.
        - Ttrue : ndarray
            True treatment vector for evaluation.
        - Xtrue : ndarray
            True covariate vector for evaluation.
        - Group.Effect : float
            The effect size parameter.
        - Covar.Effect : float
            The covariate effect size parameter.
        - R : ndarray, optional
            The rotation matrix applied if rotate=True.

    References
    ----------
    Eric W. Bridgeford, et al. "Learning Sources of Variability from High-Dimensional
    Observational Studies" arXiv (2025).
    """
    # Check input parameters
    check_in = _CheckInputs(n, p)
    check_in()

    rng = check_random_state(random_state)

    # First get base data from sigmoidal_sim with zero effect size
    result = sigmoidal_sim(
        n=n,
        p=p,
        pi=pi,
        balance=balance,
        eff_sz=0,
        covar_eff_sz=covar_eff_sz,
        alpha=alpha,
        beta=beta,
        common=common,
        a=a,
        b=b,
        err=err,
        nbreaks=nbreaks,
        rotate=False,
        random_state=rng,
    )

    Ys = result["Ys"]
    Ts = result["Ts"]

    # Add heteroskedastic noise to control group
    idx = np.where(Ts == 0)[0]
    hetero_noise = rng.normal(0, np.sqrt(2 * eff_sz), size=(len(idx), p))
    Ys[idx, :] = Ys[idx, :] + hetero_noise

    out = {
        "Ys": Ys,
        "Ts": Ts,
        "Xs": result["Xs"],
        "Eps": result["Eps"],
        "Ytrue": result["Ytrue"],
        "Ttrue": result["Ttrue"],
        "Xtrue": result["Xtrue"],
        "Group.Effect": eff_sz,
        "Covar.Effect": covar_eff_sz,
    }

    if rotate:
        # If desired, generate and apply a rotation matrix
        R = random_rotation(p, random_state=rng)
        out["Ys"] = out["Ys"] @ R.T
        out["R"] = R

    return out


CATE_SIMULATIONS = {
    "Sigmoidal": sigmoidal_sim,
    "Non-Monotone": nonmonotone_sim,
    "K-Class Sigmoidal": kclass_sigmoidal_sim,
    "Heteroskedastic Sigmoidal": heteroskedastic_sigmoidal_sim,
}


def cate_sim(sim, n, p, **kwargs):
    r"""
    Conditional ATE simulation generator.

    Takes a simulation and the required parameters, and outputs the simulated
    data matrices.

    Parameters
    ----------
    sim : str
        The name of the simulation (from the :mod:`hyppo.tools` module) that is to be generated.
    n : int
        The number of samples desired by the simulation (>= 5).
    p : int
        The number of dimensions desired by the simulation (>= 1).
    **kwargs
        Additional keyword arguements for the desired simulation.

    Returns
    -------
    x,y : ndarray of float
        Simulated data matrices.
    """
    if sim not in CATE_SIMULATIONS.keys():
        raise ValueError(
            "sim_name must be one of the following: {}".format(list(SIMULATIONS.keys()))
        )
    else:
        sim = CATE_SIMULATIONS[sim]

    return sim(n, p, **kwargs)
