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