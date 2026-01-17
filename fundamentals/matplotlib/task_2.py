import matplotlib.pyplot as plt
import numpy as np

days = np.array([1, 2, 3, 4, 5, 6, 7])
product_A = np.array([100, 150, 120, 170, 200, 180, 220])
product_B = np.array([90, 130, 110, 160, 190, 170, 210])

plt.plot(days, product_A, marker="o", color="red", label="Product A")
plt.plot(days, product_B, marker="s", color="blue", label="Product B")

plt.title("Sales Comparison")
plt.xlabel("Day")
plt.ylabel("Sales")

plt.grid()
plt.legend()

plt.show()