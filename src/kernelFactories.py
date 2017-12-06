import numpy as np
import numpy.linalg as npalg


def polynomial_kernel(P):
    def kernel(x, y):
        return (1 + np.dot(x, y)) ** P
    return kernel


def gaussian_kernel(SIGMA):
    def kernel(x, y):
        return np.exp(-npalg.norm(x - y)**2 / (2 * (SIGMA ** 2)))
    return kernel
