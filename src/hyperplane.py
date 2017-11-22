import numpy as np


class Hyperplane:
    def __init__(self, dim):
        self.W = np.ones(dim)
        self.b = 0

    def calcImplicitFunction(self, X):
        return self.W.T.dot(X) + self.b

    def calcDistance(self, X):
        return abs(self.calcImplicitFunction(X)) / np.linalg.norm(self.W)
