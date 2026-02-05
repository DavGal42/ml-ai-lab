"""
Visualize the number of orders across different product categories using a bar chart.
Store the category names and their corresponding order counts,
then use Matplotlib to create a bar plot that clearly shows how orders are distributed among categories.
Customize the chart with a title, axis labels, and rotated category labels for better readability.
"""

import matplotlib.pyplot as plt
import numpy as np

categories = ['Electronics', 'Clothing', 'Furniture', 'Toys']
orders = np.array([120, 150, 80, 60])


plt.bar(categories, orders, color="green", )

plt.title("Orders by Category")
plt.ylabel("Number of Orders")
plt.xlabel("Categories")

plt.xticks(rotation=45)

plt.show()