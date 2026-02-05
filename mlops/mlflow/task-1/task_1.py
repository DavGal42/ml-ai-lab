"""
Train a Linear Regression model on the California Housing dataset and track the experiment using MLflow:
- Load the dataset and split it into training and testing sets
- Set an MLflow experiment
- Start an MLflow run
- Train a LinearRegression model and make predictions
- Compute the Mean Squared Error (MSE) on the test set
- Log model parameters (model_type, fit_intercept), metrics (mse), and the trained model to MLflow
- Print the MSE to the console
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X, y = fetch_california_housing(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("California_linear_regression")

with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("fit_intercept", model.fit_intercept)

    mlflow.log_metric("mse", mse)

    mlflow.sklearn.log_model(model, "linear_model")

    print(f"Run finished. MSE: {mse}")