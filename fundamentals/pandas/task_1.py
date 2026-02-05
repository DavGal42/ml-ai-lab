"""
Create a Pandas Series of integers from 1 to 10 and compute basic statistics:
- Sum of all elements
- Mean value
- Maximum value
  Print the Series and the computed statistics to practice basic Pandas operations.
"""

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