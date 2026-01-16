import pandas as pd

series = pd.Series(range(1, 6))

result = (series * 10 + 5) ** 2

print(f'Series\n {series}')
print(f'\nResult\n {result}')