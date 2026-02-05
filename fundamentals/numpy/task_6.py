"""
Perform basic matrix operations using NumPy:
- Define two matrices A (2×3) and B (3×2)
- Compute the matrix multiplication A @ B
- Compute the transpose of matrix A
  Print the matrices and the results. This helps practice linear algebra operations commonly used in ML computations.
"""

import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

B = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])

product = A @ B
transpose_A = A.T

print(f'Matrix A:\n {A}')
print(f'Matrix B:\n {B}')

print(f'A @ B:\n {product}')
print(f'Transpose of A:\n {transpose_A}')