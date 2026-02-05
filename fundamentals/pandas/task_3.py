"""
Create a Pandas Series of integers from 1 to 20 and filter values using conditions:
- Extract only even numbers
- From the even numbers, keep only those greater than 10
- Print the original Series, the even numbers, and the filtered numbers
"""

import pandas as pd

series = pd.Series(range(1, 21))

even_nums = series[series % 2 == 0]

filtered_nums = even_nums[even_nums > 10]

print(f'Series\n {series}')
print(f'\nEven Numbers\n {even_nums}')
print(f'\nFiltered Numbers\n {filtered_nums}')