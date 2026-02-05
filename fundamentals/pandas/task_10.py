"""
- Load a CSV file (task_10_data.csv) into a Pandas DataFrame
- Group the data by the "department" column
- Compute the following aggregations for each group:
  * Average salary (mean)
  * Maximum salary (max)
  * Employee count (count)
- Sort the grouped results by average salary in descending order
- Print the final result
"""

import pandas as pd

df = pd.read_csv("data/task_10_data.csv")

grouped = df.groupby("department").agg(
    avg_salary=("salary", "mean"),
    max_salary=("salary", "max"),
    employee_count=("salary", "count")
)

result = grouped.sort_values(by="avg_salary", ascending=False)

print(result)