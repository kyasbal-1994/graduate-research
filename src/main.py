
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from hyperplane import Hyperplane
from svm import SVM
from reporter import Reporter

iris = datasets.load_iris()
# 先頭から100個のデータ(setosaとversicolorを抽出)
# 特徴は0番目(sepal length)と2列目(petal length)を使用
data = iris.data[0:100][:, [0, 1]]
target = iris.target[0:100]
target[target != 0] = 1
target[target == 0] = -1
# print(target)

for i in range(0, 100):
    if target[i] == -1:
        print("%s, %s" % (data[i][0], data[i][1]))
hp = Hyperplane(data[0].size)
svm = SVM(data, target, Reporter())
svm.learn(10000, 100)
svm.updateHyperplane()

print(svm.hp.W)
print(svm.hp.b)
