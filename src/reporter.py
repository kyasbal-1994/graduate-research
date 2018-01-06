from svm import SVM
import numpy.linalg as npa
import numpy as np
import matplotlib.pyplot as plt


class Reporter:
    def __init__(self, useChart, testX, testY):
        self.useChart = useChart
        self.testX = testX
        self.testY = testY

    def report(self, svm: SVM, i: float, el: float) -> str:
        if self.useChart:
            x = np.linspace(4, 10, 100)
            y = (-svm.hp.b - svm.hp.W[0] * x) / svm.hp.W[1]
            plt.scatter(svm.X[svm.T == -1][:, 0], svm.X[svm.T == -1][:, 1])
            plt.scatter(svm.X[svm.T == 1][:, 0], svm.X[svm.T == 1][:, 1])
            plt.plot(x, y, "r-")
            plt.savefig("%s.png" % (str(i)))
            plt.clf()

        return "Epoc - %s , %s\nW norm : %s (Test : %s%%)" % (i, el, npa.norm(svm.calcW()), svm.test(self.testX, self.testY) * 100)


class ConvergenceReporter:
    
    def __init__(self, testX, testY):
        self.testX = testX
        self.testY = testY
        self.maxPercentage = 0

    def report(self, svm: SVM, i: float, el: float) -> str:
        current = svm.test(self.testX, self.testY) * 100
        self.maxPercentage = current if current > self.maxPercentage else self.maxPercentage
        return "%s,%s,%s,%s(max:%s)" % (i, npa.norm(svm.calcW()),current, svm.test(svm.X, svm.T) * 100,self.maxPercentage)
