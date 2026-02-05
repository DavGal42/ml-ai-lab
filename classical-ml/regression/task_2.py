"""
Predict diamond prices using Linear Regression with a full ML pipeline:
- Load the diamonds dataset and select only numeric features
- Split the data into training and test sets
- Build a scikit-learn Pipeline including StandardScaler and LinearRegression
- Train the pipeline on the training set and make predictions on the test set
- Compute Mean Squared Error (MSE) and R² score
- Create two plots:
  * Actual vs Predicted prices
  * Residuals vs Predicted values
- Save both plots as PNG files
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("../../data/raw/diamonds.csv")

numeric_df = df.select_dtypes(include=np.number)

X = numeric_df.drop("price", axis=1)
y = numeric_df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.3f}")

plt.figure()
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Diamond Prices")
plt.savefig("actual_vs_predicted.png")
plt.close()

residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.savefig("residuals.png")
plt.close()