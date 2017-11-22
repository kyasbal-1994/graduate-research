from svm import SVM
import numpy.linalg as npa
import numpy as np
import matplotlib.pyplot as plt


class Reporter:
    def report(self, svm: SVM, i: float) -> str:
        x = np.linspace(4, 10, 100)
        y = (-svm.hp.b - svm.hp.W[0] * x) / svm.hp.W[1]
        plt.scatter(svm.X[svm.T == -1][:, 0], svm.X[svm.T == -1][:, 1])
        plt.scatter(svm.X[svm.T == 1][:, 0], svm.X[svm.T == 1][:, 1])
        plt.plot(x, y, "r-")
        plt.savefig("%s.png" % (str(i)))
        plt.clf()
        return "W norm : %s" % (npa.norm(svm.calcW()))
