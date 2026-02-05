"""
- Load the Iris dataset from scikit-learn
- Extract features (X) and target (y) arrays
- Print the feature names and target names
- Print the shape of X and y
"""

from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

print(f'Feature Names: {iris.feature_names}', end="\n")
print(f'Target Names: {iris.target_names}', end="\n")
print(f'X Shape: {X.shape}')
print(f'y Shape: {y.shape}')