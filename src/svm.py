import numpy as np
import time
from hyperplane import Hyperplane


class SVM:
    def __init__(self, X, T, reporter):
        self.alpha = np.zeros(T.size)
        self.hp = Hyperplane(X[0].size)
        self.beta = 1.0
        self.X = X
        self.T = T
        self.H = np.zeros((T.size, T.size))
        self.reporter = reporter
        for i in range(0, T.size):
            for j in range(0, T.size):
                self.H[i, j] = T[i] * T[j] * X[i].dot(X[j])

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
        eta_al = 0.0001  # update ratio of alpha
        eta_be = 0.1  # update ratio of beta
        for i in range(self.T.size):
            delta = 1 - (self.T[i] * self.X[i]).dot(self.alpha *
                                                    self.T * self.X.T).sum() - self.beta * self.T[i] * self.alpha.dot(self.T)
            self.alpha[i] += eta_al * delta
        for i in range(self.T.size):
            self.beta += eta_be * self.alpha.dot(self.T) ** 2 / 2

    def learn(self, N, stride=100):
        lastTime = time.time()
        for i in range(0, N):
            self.itr()
            if i % stride == 0:
                current = time.time()
                self.updateHyperplane()
                self.report(i, current - lastTime)
                lastTime = current

    def report(self, i, el):
        print("Epoc - %s , %s\n%s" % (i, el, self.reporter.report(self, i)))

    def Ld(self, alpha):
        return alpha.T.dot(np.ones(alpha.size)) - 1 / 2 * \
            alpha.T.dot(self.H).dot(alpha) - 1 / 2 * self.beta * \
            alpha.T.dot(self.T) * self.T.T.dot(alpha)
#
