"""
Predict passenger survival on the Titanic using Logistic Regression:
- Load the Titanic dataset and fill missing values for Age and Embarked
- Drop irrelevant columns (Name, PassengerId, Ticket, Cabin) and set features (X) and target (y)
- Encode categorical features using one-hot encoding
- Split the data into training and test sets
- Train a Logistic Regression model on the training data
- Make predictions on the test set
- Evaluate the model using accuracy, confusion matrix, and classification report
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("../../data/kaggle/Titanic-Dataset.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

X = df.drop(columns=["Survived", "Name", "PassengerId", "Ticket", "Cabin"], axis=1)
y = df["Survived"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))