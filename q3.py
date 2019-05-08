import numpy as np
import cvxpy as cp
from sacred import Experiment
import h5py
import matplotlib.pyplot as plt
import scipy.io
import os
from cvxpy.atoms.elementwise import minimum

ex = Experiment('q3-SDP')


@ex.config
def config():
    log_dir = "logs"
    X = scipy.io.loadmat('data/Ratings.mat')['X']


@ex.capture
def create_ifne(log_dir):
    if not os.path.exists(log_dir):
        print(f"Creating dir {log_dir}")
        os.mkdir(log_dir)


@ex.automain
def main(X):
    #################
    # Initialise
    #################
    create_ifne()

    #################
    # Variables
    #################
    # x = cp.Variable((4))
    idx = np.where(X > 0)
    m, n = X.shape
    r = cp.Variable()
    X_hat = cp.Variable((m, n))
    Y = cp.Variable((m, m), PSD=True)
    Z = cp.Variable((n, n), PSD=True)
    A = cp.bmat([[Y, X_hat],
                 [X_hat.T, Z]])
    print(A.shape)

    #################
    # Problem Setup
    #################
    obj = cp.Minimize(r)
    constraints = [X_hat[idx] == X[idx],
                   cp.trace(Y) + cp.trace(Z) <= 2*r,
                   X_hat >= 0,
                   A >> 0]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    #################
    # Inference
    #################
    _,s,_ = np.linalg.svd(X_hat.value)
    print(f"Status: {prob.status}")
    print(f"Value of r: {prob.value}")
    print(f"Actual rank (via np.linalg): {np.linalg.matrix_rank(X_hat.value, tol=1e-6)}")
    print(f"Singular values {s}")
    # print(f"Revenue per component: {revenue(x.value, vec=True)}")
    # print(f"Average price per unit for each activity level {revenue(x.value, vec=True) / x.value}")
    # print(f"Consumption levels: {A @ x.value}")
