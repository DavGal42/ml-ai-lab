import pandas as pd

series = pd.Series(range(1, 21))

even_nums = series[series % 2 == 0]

filtered_nums = even_nums[even_nums > 10]

print(f'Series\n {series}')
print(f'\nEven Numbers\n {even_nums}')
print(f'\nFiltered Numbers\n {filtered_nums}')