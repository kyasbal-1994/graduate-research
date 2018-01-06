import numpy as np
import time
from hyperplane import Hyperplane


class SVM:
    def __init__(self, X, T, C, kernel, reporter):
        self.alpha = np.zeros(T.size)
        self.hp = Hyperplane(X[0].size)
        self.beta = 1.0
        self.X = X
        self.T = T
        self.C = C
        self.kernel = kernel
        self.H = np.zeros((T.size, T.size))
        self.reporter = reporter
        for i in range(0, T.size):
            for j in range(0, T.size):
                self.H[i, j] = T[i] * T[j] * self.kernel(X[i], X[j])

    def calcW(self):
        return (self.alpha * self.T).T.dot(self.X)

    def calcb(self, W):
        index = self.alpha > 0
        return (self.T[index] - self.X[index].dot(W)).mean()

    def updateHyperplane(self):
        W = self.calcW()
        self.hp.W = W
        self.hp.b = self.calcb(W)

    def itr(self):
        eta_al = 0.00001  # update ratio of alpha
        eta_be = 0.01  # update ratio of beta
        self.alpha += eta_al * self.Ld2(self.alpha)
        self.alpha[self.alpha > self.C] = self.C
        self.alpha[self.alpha < 0] = 0
        for i in range(self.T.size):
            self.beta += eta_be * self.alpha.dot(self.T) ** 2 / 2

    def learn(self, N, stride=100):
        lastTime = time.time()
        for i in range(0, N):
            self.itr()
            if i % stride == 0:  # Only when needs update hyperplane to check result
                current = time.time()
                self.updateHyperplane()
                self.report(i, current - lastTime)
                lastTime = current

    def test(self, X, y):
        correct = 0
        for i in range(0, y.size):
            if self.hp.calcImplicitFunction(X[i]) * y[i] >= 0:
                correct += 1
        return correct / y.size

    def report(self, i, el):
        print("%s" % (self.reporter.report(self, i, el)))

    def Ld2(self, alpha):
        return np.ones(alpha.size) - self.H.dot(alpha) - self.beta * alpha.T.dot(self.T) * self.T
