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
            D[i, j] = LA.norm(X[i, :] - X[j, :]) # L2
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
    Y arg not needed?
    """
    def delta(u, i, j):
        #print(f"X shape: {(X[i] - X[j]).T.shape}")
        #print(f"sign shape: {(np.sign(u.T * (X[i] - X[j]))).shape}")
        return (X[i] - X[j]).T @ np.sign(u.T * (X[i] - X[j]))
    N = R_X.shape[0]
    grad_sum = 0.
    for i in range(N):
        for j in range(N):
            grad_sum += R_Y[i, j] * (delta(u, i, j)
            - delta(u, range(N), j)
            - delta(u, i, range(N))
            + delta(u, range(N), range(N)))
    return (1 / N**2) * grad_sum

X = np.random.rand(20, 1) # NxM, test with Nx1 vector of N samples
Y = np.random.rand(15, 1)
N = 10
D_X = dist_mat(X, N)
D_Y = dist_mat(Y, N)
R_X = re_centered_dist(D_X)
R_Y = re_centered_dist(D_Y)
v = dist_cov_sq(R_X, R_Y)
print(f"v: {v}")
u = np.random.rand(X.shape[0], 1)
dv = dist_cov_sq_grad(u, X, Y, R_X, R_Y)
print(f"dv: {dv}")
print(f"dv shape: {dv.shape}")
