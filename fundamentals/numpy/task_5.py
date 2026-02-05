"""
Generate 10 random numbers between 0 and 1 using NumPy with a fixed random seed for reproducibility.
Compute and print the mean and standard deviation of these numbers to practice basic statistical analysis
and understand the distribution of random data.
"""

import numpy as np

np.random.seed(100)
random_nums = np.random.rand(10)

mean = random_nums.mean()
std = random_nums.std()

print("Random numbers:", random_nums)
print("Mean:", mean)
print("Standard deviation:", std)