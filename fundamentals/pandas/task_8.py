"""
- Load a CSV file (task_8_data.csv) into a Pandas DataFrame
- Print the full DataFrame and the first 3 rows using .head()
- Compute the mean of the "age" column and the maximum of the "score" column
- Filter rows where "age" is greater than 25 and "score" is at least 88
- Print the filtered DataFrame
"""

import pandas as pd

df = pd.read_csv("data/task_8_data.csv")

print(f'{df}\n')

print(f'{df.head(3)}\n')

mean_age = df["age"].mean()
max_score = df["score"].max()
print("Mean age:", mean_age)
print("Max score:", max_score)

filtered_df = df[(df["age"] > 25) & (df["score"] >= 88)]
print(f'\n{filtered_df}')