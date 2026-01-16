import numpy as np
import pandas as pd

data = np.arange(1, 13).reshape(3, 4)
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])

second_row = df.iloc[1]
last_column = df.iloc[:, -1]

print(f'Dataframe\n {df}')
print(f'\nSecond Row\n {second_row}')
print(f'\nLast Column\n {last_column}')