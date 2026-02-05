"""
Create a NumPy array of integers from 1 to 20, then:
- Filter out only the even numbers
- From the even numbers, filter those greater than 10 Print the original array, the even numbers, and the filtered numbers.
  This practices conditional indexing and filtering in NumPy.
"""

import numpy as np

array = np.arange(1, 21)

arr_evens = array[array % 2 == 0]

gt_ten_nums = arr_evens[arr_evens > 10]

print(f'Array: {array}')
print(f'Only Evens {arr_evens}')
print(f'Filtered Numbers {gt_ten_nums}')