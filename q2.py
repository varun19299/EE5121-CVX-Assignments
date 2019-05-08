import numpy as np
import cvxpy as cp
from sacred import Experiment
import h5py
import matplotlib.pyplot as plt
import scipy.io
import os
from cvxpy.atoms.elementwise import minimum

ex = Experiment('q2-LP')


@ex.config
def config():
    log_dir = "logs"
    A = np.array([[1, 2, 0, 1],
                  [0, 0, 3, 1],
                  [0, 3, 1, 1],
                  [2, 1, 2, 5],
                  [1, 0, 3, 2]])
    c_max = np.ones((5)) * 100
    p = np.array([3, 2, 7, 6])
    p_disc = np.array([2, 1, 4, 2])
    q = np.array([4, 10, 5, 10])


@ex.capture
def revenue(x, p, p_disc, q, vec=False):
    r = np.zeros_like(x)
    # idx = np.where(x <= q)
    # r[idx] = (p * x)[idx]
    # idx = np.where(x >= q)
    # r[idx] = (p * q + p_disc * (x - q))[idx]

    if vec:
        r = np.minimum(p * x, p * q + p_disc * (x - q))
        return r
    else:
        r = 0
        for i in range(len(p)):
            r += cp.minimum(p[i] * x[i], p[i] * q[i] + p_disc[i] * (x[i] - q[i]))
        return r


@ex.capture
def create_ifne(log_dir):
    if not os.path.exists(log_dir):
        print(f"Creating dir {log_dir}")
        os.mkdir(log_dir)


@ex.automain
def main(A, c_max, p, p_disc, q):
    #################
    # Initialise
    #################
    create_ifne()

    #################
    # Variables
    #################
    x = cp.Variable((4))

    #################
    # Problem Setup
    #################
    obj = cp.Maximize(revenue(x))
    constraints = [x >= 0, A @ x <= c_max]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    #################
    # Inference
    #################
    print(f"Status: {prob.status}")
    print(f"Revenue: {prob.value}")
    print(f"Activity Levels: {x.value}")
    print(f"Revenue per component: {revenue(x.value, vec=True)}")
    print(f"Average price per unit for each activity level {revenue(x.value, vec=True) / x.value}")
    print(f"Consumption levels: {A @ x.value}")

