"""
- Load a CSV file (task_9_data.csv) into a Pandas DataFrame
- Identify missing values using .isna() and summarize missing counts per column
- Fill missing values in the "age" column with the mean and "score" column with the median
- Save the cleaned DataFrame to a new CSV file
- Print the cleaned DataFrame
"""

import pandas as pd

df = pd.read_csv("data/task_9_data.csv")

print(df.isna())

print(f'\n{df.isna().sum()}\n')

df["age"] = df["age"].fillna(df["age"].mean())
df["score"] = df["score"].fillna(df["score"].median())

df.to_csv("new_data.csv")

print(df)