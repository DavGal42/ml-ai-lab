import pandas as pd

df = pd.read_csv("data/task_9_data.csv")

print(df.isna())

print(f'\n{df.isna().sum()}\n')

df["age"] = df["age"].fillna(df["age"].mean())
df["score"] = df["score"].fillna(df["score"].median())

df.to_csv("new_data.csv")

print(df)