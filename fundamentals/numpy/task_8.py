"""
Perform multiple NumPy operations on a 5×5 matrix of random integers between 1 and 50:
- Generate the random matrix with a fixed seed for reproducibility
- Set all values greater than 25 to 0 (conditional filtering)
- Compute and print the mean and standard deviation of the matrix
- Add a 1D array [1, 2, 3, 4, 5] to the matrix using broadcasting
  Print the matrix at each step to understand the transformations.
"""

import numpy as np

np.random.seed(42)
matrix = np.random.randint(1, 51, (5, 5))

print(f'Matrix\n {matrix}')

gt_twenty_five = matrix > 25

matrix[gt_twenty_five] = 0

print(f'\nFiltered Matrix\n {matrix}')

mean = matrix.mean()
std = matrix.std()

print(f'\nMean: {mean}')
print(f'Std: {std}')

matrix += np.array([1, 2, 3, 4, 5])

print(f'\nMatrix after broadcasting addition:\n {matrix}')