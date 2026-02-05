"""
Convert a color image to grayscale and visualize both the original and grayscale images side by side.
Load the image using PIL, convert it to a NumPy array,
compute the grayscale values using the standard luminance formula (0.299 R + 0.587 G + 0.114 B),
and then plot both images using Matplotlib with appropriate titles.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('python-logo.png')
arr = np.array(img)

if arr.ndim == 3 and arr.shape[2] >= 3:
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    gray = 0.299*R + 0.587*G + 0.114*B
else:
    gray = arr

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(arr)

plt.subplot(1,2,2)
plt.title("Gray")
plt.imshow(gray, cmap='gray')

plt.show()