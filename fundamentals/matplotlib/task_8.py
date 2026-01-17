import matplotlib.pyplot as plt
import numpy as np

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
product_A = np.array([3000, 4000, 3500, 4500, 4200])
product_B = np.array([2000, 3000, 2500, 3500, 3300])

figure, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(months, product_A)
axes[0].set_title("Product A Sales")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Sales")
axes[0].grid(True)

axes[1].plot(months, product_B)
axes[1].set_title("Product B Sales")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Sales")
axes[1].grid(True)

figure.suptitle("Monthly Sales Comparison", fontsize=16)

plt.show()