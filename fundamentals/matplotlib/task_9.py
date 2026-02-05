"""
Visualize training and validation loss over multiple epochs to monitor model performance.
Store the loss values for training and validation, then create subplots to show each curve separately.
Add titles, axis labels, grids, an overall figure title, and save the figure as a high-resolution PNG file.
"""

import matplotlib.pyplot as plt
import numpy as np

epochs = np.array([1, 2, 3, 4, 5])
train_loss = np.array([0.9, 0.7, 0.5, 0.4, 0.35])
val_loss = np.array([1.0, 0.8, 0.6, 0.55, 0.5])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs, train_loss)
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True)

axes[1].plot(epochs, val_loss)
axes[1].set_title("Validatio Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].grid(True)

fig.suptitle("Model Loss Curves", fontsize=14)

plt.tight_layout()

plt.savefig("loss_curves.png", dpi=300)

plt.show()

