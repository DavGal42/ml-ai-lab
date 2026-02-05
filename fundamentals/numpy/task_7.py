"""
Demonstrate NumPy broadcasting by adding a 1D array to a 2D matrix:
- Define a 2×3 matrix and a 1D array of length 3
- Add the array to the matrix using broadcasting
- Print the matrix, array, and the resulting matrix
  This helps understand how NumPy automatically aligns shapes for element-wise operations.
"""

import numpy as np

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

array = np.array([10, 20, 30])

result = matrix + array

print(f'Matrix\n {matrix}')
print(f'Array: {array}')
print(f'Result\n {result}')