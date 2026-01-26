import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data

X_first_two = X[:, :2]

print("Shape of X_first_two:", X_first_two.shape)
print("First 5 rows:")
print(X_first_two[:5])

means = np.mean(X_first_two, axis=0)
mins = np.min(X_first_two, axis=0)
maxs = np.max(X_first_two, axis=0)

print("\nStatistics:")
print("Mean:", means)
print("Min:", mins)
print("Max:", maxs)