"""
- Load the Iris dataset from scikit-learn
- Split it into training and test sets with stratification (stratify=y)
- Print the mean and standard deviation of the training and test sets before scaling
- Apply StandardScaler to both sets
- Print the mean and standard deviation after scaling
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Original training set statistics:")
print("Mean:", np.mean(X_train, axis=0))
print("Std:", np.std(X_train, axis=0))

print("\nOriginal test set statistics:")
print("Mean:", np.mean(X_test, axis=0))
print("Std:", np.std(X_test, axis=0))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

print("\nScaled training set statistics:")
print("Mean:", np.mean(X_train_scaled, axis=0))
print("Std:", np.std(X_train_scaled, axis=0))

print("\nScaled test set statistics:")
print("Mean:", np.mean(X_test_scaled, axis=0))
print("Std:", np.std(X_test_scaled, axis=0))