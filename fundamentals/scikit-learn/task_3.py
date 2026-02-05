"""
- Load the Iris dataset and get X (features) and y (target)
- Convert X into a Pandas DataFrame with proper feature names
- Add the target column (y) as "target"
- Print the first 5 rows, shape, and column names of the DataFrame
"""

import pandas as pd
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

df = pd.DataFrame(X, columns=load_iris().feature_names)

df["target"] = y

print("First 5 rows: ")
print(df.head())

print("\nShape of DataFrame:", df.shape)
print("\nColumns:")
print(df.columns)