"""
Create a simple data visualization using Matplotlib to show how sales change over a 7-day period.
Use NumPy arrays to store the days and sales values, then plot a line chart with markers, axis labels,
and a title to clearly display the trend over time.
"""

import matplotlib.pyplot as plt
import numpy as np

days = np.array([1, 2, 3, 4, 5, 6, 7])
sales = np.array([100, 150, 120, 170, 200, 180, 220])

plt.title("Sales over 7 days")

plt.plot(days, sales, marker="o", color="green")
plt.xlabel("Day")
plt.ylabel("Sales")

plt.show()