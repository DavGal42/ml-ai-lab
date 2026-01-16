from numpy import mean
import pandas as pd

series = pd.Series(range(1, 11))

sum = series.sum()
mean = series.mean()
max = series.max()

print(f'Series\n {series}')
print(f'\nSum: {sum} ')
print(f'Mean: {mean} ')
print(f'Max: {max} ')