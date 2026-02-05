"""
Create a NumPy array of integers from 1 to 12, reshape it into a 3×4 matrix, and practice indexing:
- Print the entire matrix
- Print the second row
- Print the last column
This helps understand how to manipulate and access elements in multidimensional arrays.
"""

import numpy as np

array = np.arange(1, 13)

print(array)

matrix = array.reshape(3, 4)

print(matrix)

print(matrix[1])

print(matrix[:,-1])