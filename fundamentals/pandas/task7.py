import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame(np.random.randint(1, 31, (5, 3)), columns=['A', 'B', 'C'])

df['total'] = df[['A', 'B', 'C']].sum(axis=1)
df['mean'] = df[['A', 'B', 'C']].mean(axis=1)

filtered_df = df[df['total'] > 50]

print("DataFrame:\n", df)
print("\nFiltered DataFrame (total > 50):\n", filtered_df)