import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()

X = iris.data[:, 2]
y = iris.target

model = LogisticRegression(multi_class='ovr', max_iter=200)
model.fit(X, y)

