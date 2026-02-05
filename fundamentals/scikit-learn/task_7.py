"""
- Load the Iris dataset from scikit-learn
- Split the features X and target y into training and test sets using train_test_split
  * Test size: 20%
  * Set a random seed (random_state=42) for reproducibility
- Print the shapes of X_train, X_test, y_train, and y_test
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)