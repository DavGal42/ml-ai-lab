"""
Visualize how total sales are distributed across different product categories using a pie chart.
Store the sales values and category labels, then create a pie chart that shows each category’s percentage contribution to total sales.
Highlight the largest category using an exploded slice and customize the chart with colors and percentage labels.
"""

import matplotlib.pyplot as plt
import numpy as np

categories = ['Electronics', 'Clothing', 'Furniture', 'Toys']
sales = np.array([5000, 3000, 2000, 1000])

plt.pie(
    sales,
    labels=categories,
    autopct='%1.1f%%',
    explode=[0.1, 0, 0, 0],
    colors=['orange', 'blue', 'green', 'red'],
)

plt.title("Sales Distribution by Category")

plt.show()