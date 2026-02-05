"""
Compare the sales performance of two different products over a 7-day period using data visualization.
Store the daily sales data for each product in NumPy arrays, then plot both datasets on the same line chart using different markers,
colors, and labels. Add a title, axis labels, a legend, and a grid to make the comparison clear and easy to read.
"""

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