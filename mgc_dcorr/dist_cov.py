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
    Are X and Y same length?
    """
    N = R_X.shape[0] # if R square and same length
    v_sum = 0.
    for i in range(N):
        for j in range(N):
            v_sum += R_X[i, j] * R_Y[i, j]
    return (1 / N**2) * v_sum

def dist_cov_sq_grad(u, X, R_Y):
    """
    Gradient for use in projected stochastic gradient descent optimization
    """
    def delta(u, i, j):
        sign_term = np.squeeze(np.sign((X[i] - X[j]) @ u))
        #print(f"X shape: {(X[i] - X[j]).T.shape}")
        #print(f"sign term: {sign_term}")
        if sign_term.shape == ():
            return (X[i] - X[j]).T * sign_term.item() # singleton to scaler
        else:
            return (X[i] - X[j]).T @ sign_term
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

def clamp_u(u):
    norm = LA.norm(u)
    if norm > 1:
        return  u / norm
    else:
        return u

def optim_u_gd(u, X, R_Y, lr, epsilon):
    """
    Gradient ascent for v^2 with respect to u
    """
    u_opt = np.copy(u)
    R_X_u = re_centered_dist_u(u, X)
    v = dist_cov_sq(R_Y, R_X_u)
    while True:
        grad = dist_cov_sq_grad(u_opt, X, R_Y)
        u_opt += lr * grad # "+=": gradient ascent
        u_opt = clamp_u(u_opt)
        R_X_u_opt = re_centered_dist_u(u_opt, X)
        v_opt = dist_cov_sq(R_Y, R_X_u_opt)
        delta = LA.norm(v_opt- v)
        if delta <= epsilon:
            break
        else:
            v = v_opt
    return u_opt

def k_test():
    """
    Test if U[:, k] is significant with respect to U[:, 1:k-1]
    Permutation test not needed for single dataset X
    """

def proj_U(X, U, k):
    """
    Project X onto the orthogonal subspace of k dim of U
    """
    q, _ = LA.qr(U[:, :k])
    #X_proj = np.sum(X * U[:, :k].T, axis=1) # vectorized dot
    X_proj = np.zeros_like(X) # looped proj
    for n in range(X_proj.shape[0]):
        for k_i in range(k):
            X_proj[n] = X_proj[n] + (np.dot(X[n], q[:, k_i]) / np.dot(q[:, k_i], q[:, k_i])) * q[:, k_i]
    return X_proj

def dca(X, Y, lr, epsilon):
    """
    Perform DCA dimensionality reduction on X
    Single dataset X
    """
    k = 0
    U = np.zeros_like(X.T) # kmax is num of X features?
    X_proj = np.copy(X)
    R_Y = re_centered_dist(Y)
    while True:
        U[:, k] = clamp_u(np.random.rand(X.shape[1], 1))
        R_X_proj = re_centered_dist(X_proj)
        u_opt = optim_u_gd(U[:, k], X_proj, Y, R_X_proj, R_Y, lr, epsilon)
        U[:, k] = u_opt
        if k_test(U, k):
            X_proj = proj_U(X_proj, U[:, :k])
            k += 1
        else:
            break
    return X_proj
