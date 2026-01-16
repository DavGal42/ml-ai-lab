import numpy as np

np.random.seed(100)
random_nums = np.random.rand(10)

mean = random_nums.mean()
std = random_nums.std()

print("Random numbers:", random_nums)
print("Mean:", mean)
print("Standard deviation:", std)