
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import kernelFactories as kf
from sklearn.model_selection import train_test_split
from sklearn import datasets
from hyperplane import Hyperplane
from svm import SVM
import reporter as r

targetIndex = 6

# IRIS
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# y[y != targetIndex] = 1
# y[y == targetIndex] = -1
# WINES
# X, y = datasets.load_wine(return_X_y=True)
# y[y != targetIndex] = 1
# y[y == targetIndex] = -1
# CANCER
# X, y = datasets.load_breast_cancer(return_X_y=True)
# y[y != targetIndex] = 1
# y[y == targetIndex] = -1
# DIGITS
# X, y = datasets.load_digits(return_X_y=True)
# y[y != targetIndex] = 1
# y[y == targetIndex] = -1
X = []
y = []
glassDataset = 272
soloDataset = 318
for i in range(1,glassDataset):
    fName = "./screened-glasses/%s.jpg" % (i)
    n = np.ndarray.flatten(np.array(Image.open(fName).convert('L'), 'f' ))/255
    X.append(n)
    y.append(1)
for i in range(1,soloDataset):
    fName = "./screened-solo/%s.JPG" % (i)
    n = np.ndarray.flatten(np.array(Image.open(fName).convert('L'), 'f' ))/255
    X.append(n)
    y.append(-1)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

svm = SVM(X_train, y_train, 10, kf.gaussian_kernel(100),
          r.ConvergenceReporter(X_test, y_test))
svm.learn(100000, 1000)
svm.updateHyperplane()
print(svm.hp.W)
print(svm.hp.b)
