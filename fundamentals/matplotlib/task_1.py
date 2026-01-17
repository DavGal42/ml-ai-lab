import matplotlib.pyplot as plt
import numpy as np

days = np.array([1, 2, 3, 4, 5, 6, 7])
sales = np.array([100, 150, 120, 170, 200, 180, 220])

plt.title("Sales over 7 days")

plt.plot(days, sales, marker="o", color="green")
plt.xlabel("Day")
plt.ylabel("Sales")

plt.show()