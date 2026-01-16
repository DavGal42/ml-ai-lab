import numpy as np

array = np.arange(1, 21)

arr_evens = array[array % 2 == 0]

gt_ten_nums = arr_evens[arr_evens > 10]

print(f'Array: {array}')
print(f'Only Evens {arr_evens}')
print(f'Filtered Numbers {gt_ten_nums}')