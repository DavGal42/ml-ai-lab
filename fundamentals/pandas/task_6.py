"""
Create a 6×4 Pandas DataFrame of random integers between 1 and 50 and filter rows based on multiple conditions:
- Column 'A' values greater than 25
- Column 'C' values less than 20
  Print both the original DataFrame and the filtered DataFrame.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame(
    np.random.randint(1, 51, (6, 4)),
    columns=['A', 'B', 'C', 'D']
)

filtered_df = df[(df['A'] > 25) & (df['C'] < 20)]

print("Original DataFrame:\n", df)
print("\nFiltered DataFrame:\n", filtered_df)