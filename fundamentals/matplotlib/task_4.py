import matplotlib.pyplot as plt
import numpy as np

categories = ['Electronics', 'Clothing', 'Furniture', 'Toys']
sales = np.array([5000, 3000, 2000, 1500])


plt.barh(categories, sales, color="blue")

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")

plt.grid(axis="x")

plt.show()