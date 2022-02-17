import numpy as np
from numpy import linalg as LA

def dist_mat(X, N):
    """
    First N samples of vector X to distance matrix D
    """
    #N = len(X)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = LA.norm(X[i] - X[j]) # L2
    return D

def re_centered_dist(D):
    """
    Distance matrix D to re-centered distance matrix R
    """
    N = D.shape[0] # D should be square NxN
    R = np.zeros_like(D)
    for i in range(N):
        for j in range(N):
            R[i, j] = D[i, j] - np.mean(D[:, j]) - np.mean(D[i, :]) + np.mean(D)
    return R

def re_centered_dist_u(u, X, N):
    return  re_centered_dist(dist_mat(X @ u, N))

def dist_cov_sq(R_X, R_Y):
    """
    Are X and Y same length?
    What is N? N is some subset < len(X) and len(Y)?
    """
    N = R_X.shape[0] # if R square and same length
    v_sum = 0.
    for i in range(N):
        for j in range(N):
            v_sum += R_X[i, j] * R_Y[i, j]
    return (1 / N**2) * v_sum

def dist_cov_sq_grad(u, X, Y, R_X, R_Y):
    """
    Gradient for use in projected stochastic gradient descent optimization
    Some args not needed?
    """
    def delta(u, i, j):
        #print(f"X shape: {(X[i] - X[j]).T.shape}")
        #print(f"sign shape: {np.sign((X[i] - X[j]) @ u).shape}")
        return (X[i] - X[j]).T @ np.sign((X[i] - X[j]) @ u)
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

def optim_u_gd(u, X, Y, R_X, R_Y, lr, num_iter):
    """
    Gradient ascent for v^2 with respect to u
    """
    u_opt = np.copy(u)
    for _ in range(num_iter):
        grad = dist_cov_sq_grad(u_opt, X, Y, R_X, R_Y)
        u_opt += lr * grad # "+=": gradient ascent
        u_opt = clamp_u(u_opt)
    return u_opt