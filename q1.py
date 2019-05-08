import numpy as np
import cvxpy as cp
from sacred import Experiment
import h5py
import matplotlib.pyplot as plt
import scipy.io
import os

ex = Experiment('q1-SOCP')


@ex.config
def config():
    y = scipy.io.loadmat('data/piecewise_constant_data.mat')['y']
    n = y.shape[0]
    log_dir = "logs"


@ex.capture
def get_A(n):
    l = [-1, 1]
    A = np.zeros((n - 1, n))
    for i in range(n - 1):
        row = [0] * i + l + [0] * (n - i - len(l))
        A[i] = row

    return np.array(A)


@ex.capture
def create_ifne(log_dir):
    if not os.path.exists(log_dir):
        print(f"Creating dir {log_dir}")
        os.mkdir(log_dir)


@ex.automain
def main(y, n, log_dir):
    #################
    # Initialise
    #################
    create_ifne()

    #################
    # Variables
    #################
    x = cp.Variable((n, 1))
    A = get_A()
    print(A.shape)

    #################
    # Problem Setup
    #################
    obj = cp.Minimize(cp.norm((x - y), 2))
    constraints = [cp.norm(A @ x, 1) <= 5.3825]
    prob = cp.Problem(obj, constraints)
    prob.solve()

    #################
    # Inference
    #################
    diffs = (A @ x.value).flatten()
    jumps = np.zeros_like(diffs)
    idx = np.where(np.abs(diffs) > 1e-2)
    jumps[idx] = diffs[idx]
    jumps = np.linalg.norm(jumps, 0)

    print(np.max(np.abs(jumps)))
    print(f"Status: {prob.status}")
    print(f"L2 Error {prob.value}")
    print(f"L1 Constraint Value {np.linalg.norm(diffs, 1)}")
    print(f"L1 Constraint Violation {prob.constraints[0].violation()}")
    print(f"No of jumps {jumps}")
    plt.plot(x.value)
    plt.plot(y, alpha=0.6)
    plt.legend(['Estimated x', 'Noisy y'])
    plt.xlabel("x")
    plt.ylabel("Value")
    plt.title(f"L2 Error {prob.value:.3f} | Jumps {jumps} | L1 Constraint Value {np.linalg.norm(diffs, 1):.3f}")
    plt.savefig(f"{log_dir}/piecewise-estimate.png")
    plt.show()
