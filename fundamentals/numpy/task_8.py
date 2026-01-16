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