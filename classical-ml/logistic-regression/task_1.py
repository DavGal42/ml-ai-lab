"""
Train a Logistic Regression model on the Iris dataset and evaluate its performance:
- Load the Iris dataset and split it into training and test sets
- Initialize a LogisticRegression model (increase max_iter to ensure convergence)
- Train the model on the training set
- Make predictions on the test set
- Compute and print the accuracy score
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')