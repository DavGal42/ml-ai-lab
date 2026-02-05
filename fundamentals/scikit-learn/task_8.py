"""
- Load the Iris dataset from scikit-learn
- Split the dataset into training and test sets using train_test_split with stratification (stratify=y)
  * Test size: 20%
  * Set a random seed (random_state=42)
- Print the class counts in the training and test sets to verify the stratification
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set class counts:", np.bincount(y_train))
print("Test set class counts:", np.bincount(y_test))