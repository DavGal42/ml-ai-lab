import numpy as np

# array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
array = np.arange(1, 11)

arr_sum = array.sum()
arr_mean = array.mean()
arr_max = array.max()

print(f'Array: {array}')
print(f'Sum: {arr_sum}')
print(f'Mean: {arr_mean}')
print(f'Max: {arr_max}')