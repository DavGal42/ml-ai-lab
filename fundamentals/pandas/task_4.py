"""
Create a Pandas Series with integers [1, 2, 3, 4, 5] and apply a vectorized transformation:
- Multiply each element by 10
- Add 5
- Square the result
  Print the original Series and the transformed Series.
"""

import pandas as pd

series = pd.Series(range(1, 6))

result = (series * 10 + 5) ** 2

print(f'Series\n {series}')
print(f'\nResult\n {result}')