"""
Predict students' math scores using Linear Regression on the Students Performance dataset:
- Load the dataset and encode categorical features using one-hot encoding
- Split the data into features (X) and target (y), then into training and test sets
- Train a Linear Regression model on the training data
- Make predictions on the test set
- Evaluate the model using RMSE (Root Mean Squared Error) and R² score
- Print the evaluation metrics to assess model performance
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("../../data/kaggle/StudentsPerformance.csv")
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop(columns=["math score", "reading score", "writing score"], axis=1)
y = df["math score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse}')
print(f'R2: {r2}')