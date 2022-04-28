from numba import njit, prange
import numpy as np
from numpy import linalg as LA

@njit(parallel=True)
def mean_numba_axis0(A):
    res = []
    for i in prange(A.shape[1]):
        res.append(A[:, i].mean())
    return np.array(res)

@njit(parallel=True)
def mean_numba_axis1(A):
    res = []
    for i in prange(A.shape[0]):
        res.append(A[i, :].mean())
    return np.array(res)

@njit(parallel=True)
def dist_mat(X):
    """
    Vector X to distance matrix D
    """
    N = len(X)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = LA.norm(X[i] - X[j]) # L2
    return D

@njit(parallel=True)
def dist_mat_vec(X):
    """
    Vector X: (u^T X) to distance matrix D
    For distance matrix of u^T X
    TODO: Norm of scalar is identity?
    """
    N = len(X)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = np.abs(X[i] - X[j]) # L2
    return D

@njit(parallel=True)
def dist_mat_u(u, X):
    """
    TODO: X @ u is vector, not matrix?
    """
    u_X = X @ u
    D_u = dist_mat_vec(u_X)
    return D_u

@njit(parallel=True)
def re_centered_dist(D):
    """
    Distance matrix D to re-centered distance matrix R
    """
    c_mean = mean_numba_axis0(D)
    r_mean = mean_numba_axis1(D)
    m_mean = np.mean(D)
    N = D.shape[0] # D should be square NxN, where N is len(X)
    R = np.zeros_like(D)
    for i in range(N):
        for j in range(N):
            R[i, j] = D[i, j]
            - c_mean[j]
            - r_mean[i]
            + m_mean
    return R

@njit(parallel=True)
def dist_cov_sq(R_X, R_Y):
    """
    TODO: replace with hyppo implementation
    Uses re-centered distance covariance matrices
    """
    v_sum = np.sum(R_X * R_Y)
    N = R_X.shape[0] # R must be square and same length
    return v_sum / N**2

@njit(parallel=True)
def proj_U(X, U, k):
    """
    Project X onto the orthogonal subspace of k dim of U
    """
    q, _ = LA.qr(U[:, :k])
    #X_proj = np.sum(X * U[:, :k+1].T, axis=1) # vectorized dot
    X_proj = np.zeros_like(X) # looped proj
    for n in range(X_proj.shape[0]):
        for k_i in range(k):
            X_proj[n] = X_proj[n] + ((X[n] @ q[:, k_i]) / (q[:, k_i] @ q[:, k_i])) * q[:, k_i]
    return X_proj

@njit(parallel=True)
def dca(X, Y, K):
    """
    Perform DCA dimensionality reduction on X
    Single dataset X
    K is desired dim for reduction of X
    """
    # N = X.shape[0]
    P = X.shape[1]
    k = 0
    v = np.zeros(X.shape[1])
    U = np.zeros_like(X.T)
    X_proj = np.copy(X)
    D_Y = dist_mat(Y)
    R_Y = re_centered_dist(D_Y)
    for k in range(0, K):
        v_feat = np.zeros(P)
        for i in range(P):
            u = X[:, i]
            D_u = dist_mat_u(u, X)
            R_X_u = re_centered_dist(D_u)
            v_feat[i] = dist_cov_sq(R_Y, R_X_u)
        idx_max = np.argmax(v_feat)
        u_opt = X[:, idx_max]
        U[:, k] = u_opt
        v[k] = v_feat[idx_max]
        X_proj = proj_U(X_proj, U, k+1) # then inc k, unnecessary if this is last k
    return U[:, :k+1], v[:k+1]

@njit(parallel=True)
def dist_mat_vec_diff(X):
    """
    Not needed, X not X @ u used for diff
    Vector X: (u^T X) to distance matrix D
    For distance matrix of u^T X
    Norm of scalar is identity
    TODO: Derivative of norm of scalar is 1?
    """
    N = X.shape[0]
    P = 1 # X.shape[1]
    D_diff = np.zeros((N, N, P))
    for i in range(N):
        for j in range(N):
            diff = X[i] - X[j]
            if diff == 0: #.all()
                D_diff[i, j] = diff
            else:
                D_diff[i, j] = diff / diff
    return D_diff

@njit(parallel=True)
def dist_mat_u_diff(u, X):
    """
    Not needed, X not X @ u used for grad
    TODO: X @ u is vector, not matrix?
    """
    u_X = X @ u
    D_u_diff = dist_mat_vec_diff(u_X)
    return D_u_diff

@njit(parallel=True)
def dist_cov_sq_grad(u, X, R_Y):
    """
    Gradient for use in projected gradient descent optimization
    """
    def delta(X, u, i, j):
        sign_term = np.sign((X[i] - X[j]) @ u)
        return (X[i] - X[j]) * sign_term
    def delta_axis0(X):
        N = X.shape[0]
        P = X.shape[1]
        res = np.zeros((N, P))
        for i in range(N):
            sign_term = np.sign((X - X[i]) @ u)
            res[i] = (X - X[i]).T @ sign_term
        return res
    def delta_axis1(X):
        N = X.shape[0]
        P = X.shape[1]
        res = np.zeros((N, P))
        for i in range(N):
            sign_term = np.sign((X[i] - X) @ u)
            res[i] = (X[i] - X).T @ sign_term
        return res
    c_delta = delta_axis0(X)
    r_delta = delta_axis1(X)
    N = X.shape[0]
    P = X.shape[1]
    grad_sum = np.zeros(P)
    for i in range(N):
        for j in range(N):
            grad_sum = grad_sum + R_Y[i, j] * (
                delta(X, u, i, j)
                - c_delta[j]
                - r_delta[i]
            )
    return (1 / N**2) * grad_sum.T

def dist_cov_sq_grad_stochastic(u, X, R_Y, sto_sample):
    """
    Gradient for use in projected stochastic gradient descent optimization
    """
    def delta(u, i, j):
        sign_term = np.squeeze(np.sign((X[i] - X[j]) @ u))
        #print(f"X shape: {(X[i] - X[j]).T.shape}")
        #print(f"sign term: {sign_term}")
        return np.dot((X[i] - X[j]).T, sign_term)
    N = R_Y.shape[0]
    grad_sum = 0.
    for j in range(N):
        grad_sum += R_Y[j] * (
            delta(u, sto_sample, j)
            - delta(u, range(N), j)
            - delta(u, sto_sample, range(N))
            + delta(u, range(N), range(N))
        )
    return (1 / N**2) * grad_sum.T

@njit(parallel=True)
def normalize_u(u):
    norm = LA.norm(u)
    return  u / norm

@njit(parallel=True)
def optim_u_gd(u, X, R_Y, lr, epsilon):
    """
    Gradient ascent for v^2 with respect to u
    TODO: Regularization?
    """
    D_u = dist_mat_u(u, X)
    R_X_u = re_centered_dist(D_u)
    v = dist_cov_sq(R_Y, R_X_u)
    u_opt = np.copy(u)
    #iter_ct = 0
    while True:
        #iter_ct += 1
        #print(iter_ct)
        grad = dist_cov_sq_grad(u_opt, X, R_Y)
        u_opt = u_opt - lr * grad # "+=": gradient ascent
        u_opt = normalize_u(u_opt)
        D_u = dist_mat_u(u_opt, X)
        R_X_u_opt = re_centered_dist(D_u)
        v_opt = dist_cov_sq(R_Y, R_X_u_opt)
        delta = np.abs(v_opt - v) #MSE
        if delta <= epsilon:
            break
        else:
            v = v_opt
    return u_opt, v_opt

def optim_u_gd_stochastic(u, X, R_Y, lr, epsilon):
    """
    Stochastic gradient ascent for v^2 with respect to u
    TODO: Regularization?
    """
    sample_ct = X.shape[0]
    R_X_u = re_centered_dist_u(u, X)
    v = dist_cov_sq(R_Y, R_X_u)
    u_opt = np.copy(u)
    while True:
        sto_sample = np.random.randint(0, sample_ct)
        grad = dist_cov_sq_grad_stochastic(u_opt, X, R_Y[sto_sample], sto_sample)
        u_opt += lr * grad # "+=": gradient ascent
        u_opt = normalize_u(u_opt)
        R_X_u_opt = re_centered_dist_u(u_opt, X)
        v_opt = dist_cov_sq(R_Y, R_X_u_opt)
        delta = np.mean(np.square(v_opt - v)) #MSE
        if delta <= epsilon:
            break
        else:
            v = v_opt
    return u_opt, v_opt

@njit(parallel=True)
def dca_grad_learn(X, Y, K, lr=1e-1, epsilon=1e-5):
    """
    Perform DCA dimensionality reduction on X
    Single dataset X
    K is desired dim for reduction of X
    Use gradient ascent approach to learn representation U
    """
    k = 0
    v = np.zeros(X.shape[1])
    U = np.zeros_like(X.T)
    X_proj = np.copy(X)
    D_Y = dist_mat(Y)
    R_Y = re_centered_dist(D_Y)
    for k in range(0, K):
        u_init = normalize_u(np.random.rand(X.shape[1]))
        u_opt, v_opt = optim_u_gd(u_init, X_proj, R_Y, lr, epsilon)
        U[:, k] = u_opt
        v[k] = v_opt
        X_proj = proj_U(X_proj, U, k+1) # then inc k, unnecessary if this is last k
    return U[:, :k+1], v[:k+1]
