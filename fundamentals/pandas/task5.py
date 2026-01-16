import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame(np.random.randint(1, 21, (4, 3)), columns=['A', 'B', 'C'])

column_sum = df.sum()
column_mean = df.mean()
column_std = df.std()

print(f'DataFrame\n {df}')
print(f'\nSum per column\n {column_sum}')
print(f'\nMean per column\n {column_mean}')
print(f'\nStd per column\n {column_std}')