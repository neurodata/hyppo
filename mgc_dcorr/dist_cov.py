import numpy as np
from numpy import linalg as LA

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

def re_centered_dist(D):
    """
    Distance matrix D to re-centered distance matrix R
    """
    N = D.shape[0] # D should be square NxN, where N is len(X)
    R = np.zeros_like(D)
    for i in range(N):
        for j in range(N):
            R[i, j] = D[i, j] - np.mean(D[:, j]) - np.mean(D[i, :]) + np.mean(D)
    return R

def re_centered_dist_u(u, X):
    return  re_centered_dist(dist_mat(X @ u))

def dist_cov_sq(R_X, R_Y):
    """
    Uses re-centered distance covariance matrices
    """
    v_sum = np.sum(R_X * R_Y)
    N = R_X.shape[0] # R must be square and same length
    return v_sum / N**2

def dist_cov_sq_grad(u, X, R_Y):
    """
    Gradient for use in projected gradient descent optimization
    """
    def delta(u, i, j):
        sign_term = np.squeeze(np.sign((X[i] - X[j]) @ u))
        #print(f"X shape: {(X[i] - X[j]).T.shape}")
        #print(f"sign term: {sign_term}")
        return np.dot((X[i] - X[j]).T, sign_term)
    N = R_Y.shape[0]
    grad_sum = 0.
    for i in range(N):
        for j in range(N):
            grad_sum += R_Y[i, j] * (
                delta(u, i, j)
                - delta(u, range(N), j)
                - delta(u, i, range(N))
                + delta(u, range(N), range(N))
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

def normalize_u(u):
    norm = LA.norm(u)
    return  u / norm

def optim_u_gd(u, X, R_Y, lr, epsilon):
    """
    Gradient ascent for v^2 with respect to u
    TODO: Regularization?
    """
    R_X_u = re_centered_dist_u(u, X)
    v = dist_cov_sq(R_Y, R_X_u)
    u_opt = np.copy(u)
    #iter_ct = 0
    while True:
        #iter_ct += 1
        #print(iter_ct)
        grad = dist_cov_sq_grad(u_opt, X, R_Y)
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
        grad = dist_cov_sq_grad_stochastic(u_opt, X, R_Y[sto_sample], sto_sample) # TODO: rewrite this for single sample?
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

def k_test(v, v_opt, k, p=.1):
    """
    Test if U[:, k] is significant with respect to U[:, 1:k-1]
    Permutation test not needed for single dataset X
    TODO: Viable for single dataset?
    TODO: Always fails for low k?
    """
    if k == 0:
        return True
    else:
        if sum(v_opt > v[:k]) / k > 1 - p: # k is also len(v[:k])
            return True
        else:
            return False

def proj_U(X, U, k):
    """
    Project X onto the orthogonal subspace of k dim of U
    """
    q, _ = LA.qr(U[:, :k])
    #X_proj = np.sum(X * U[:, :k+1].T, axis=1) # vectorized dot
    X_proj = np.zeros_like(X) # looped proj
    for n in range(X_proj.shape[0]):
        for k_i in range(k):
            X_proj[n] = X_proj[n] + (np.dot(X[n], q[:, k_i]) / np.dot(q[:, k_i], q[:, k_i])) * q[:, k_i]
    return X_proj

def dca(X, Y, K=None, lr=1e-1, epsilon=1e-5):
    """
    Perform DCA dimensionality reduction on X
    Single dataset X
    K is desired dim for reduction of X
    """
    k = 0
    v = np.zeros(X.shape[1])
    U = np.zeros_like(X.T)
    X_proj = np.copy(X)
    D_Y = dist_mat(Y)
    R_Y = re_centered_dist(D_Y)
    for k in range(0, K):
        u_init = normalize_u(np.random.rand(X.shape[1]))
        u_opt, v_opt = optim_u_gd_stochastic(u_init, X_proj, R_Y, lr, epsilon)
        if K is not None or k_test(v, v_opt, k):
            U[:, k] = u_opt
            v[k] = v_opt
            X_proj = proj_U(X_proj, U, k+1) # then inc k, unnecessary if this is last k
        else:
            break
    return U[:, :k+1], v[:k+1]
