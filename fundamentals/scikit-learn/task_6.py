import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()

X = iris.data

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

means = np.mean(X_scaled, axis=0)
stds = np.std(X_scaled, axis=0)

print("Means after scaling:")
print(means)

print("\nStandard deviations after scaling:")
print(stds)