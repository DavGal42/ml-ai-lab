"""
Visualize the combined monthly sales of two products using a stacked bar chart.
Store the monthly sales data for Product A and Product B, then plot a stacked bar chart where Product B is stacked on top of Product A.
Add a title, axis labels, and a legend to clearly show each product’s contribution to total monthly sales.
"""

import matplotlib.pyplot as plt
import numpy as np

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
product_A = np.array([3000, 4000, 3500, 4500, 4200])
product_B = np.array([2000, 3000, 2500, 3500, 3300])

plt.bar(months, product_A, color="orange", label="Product A")
plt.bar(months, product_B, bottom=product_A, color="blue", label="Product B")

plt.title("Monthly Sales by Product")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.legend()

plt.show()