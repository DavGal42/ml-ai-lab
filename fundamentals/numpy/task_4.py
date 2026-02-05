"""
Create a NumPy array of integers [1, 2, 3, 4, 5].
Apply a vectorized operation on the array: multiply each element by 10, add 5, and then square the result.
Print both the original array and the transformed array. This helps practice element-wise computations using NumPy.
"""

import numpy as np

array = np.array([1, 2, 3, 4, 5])

new_arr = (array * 10 + 5) ** 2

print(f'Array: {array}')
print(f'New Array: {new_arr}')