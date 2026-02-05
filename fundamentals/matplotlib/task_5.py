"""
Visualize monthly sales and profit on the same chart to compare overall revenue with profitability trends.
Store monthly sales and profit data, then use a bar chart to represent sales and a line chart to represent profit.
Add a title, axis labels, and a legend to clearly distinguish between the two metrics.
"""

import matplotlib.pyplot as plt
import numpy as np

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales = np.array([5000, 7000, 6000, 8000, 7500])
profit = np.array([1200, 1500, 1300, 1800, 1600])

plt.bar(months, sales, color="orange", label="Sales")
plt.plot(months, profit, color="red", marker="o", label="Profit")

plt.title("Monthly Sales & Profit")
plt.xlabel("Month")
plt.ylabel("Amount")

plt.legend()

plt.show()