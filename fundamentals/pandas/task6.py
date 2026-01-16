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