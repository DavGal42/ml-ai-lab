import pandas as pd

df = pd.read_csv("data/task_10_data.csv")

grouped = df.groupby("department").agg(
    avg_salary=("salary", "mean"),
    max_salary=("salary", "max"),
    employee_count=("salary", "count")
)

result = grouped.sort_values(by="avg_salary", ascending=False)

print(result)