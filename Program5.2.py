import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Upload image
uploaded = files.upload()

# Get uploaded image path
image_path = list(uploaded.keys())[0]

# Read image (COLOR)
color_img = cv2.imread(image_path)

if color_img is None:
    raise ValueError("Failed to load image!")

# Convert BGR → RGB for display
color_img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

# Convert COLOR → GRAYSCALE for Otsu
gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

# Apply Otsu Thresholding
threshold_value, binary_img = cv2.threshold(
    gray_img,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

print("Optimal Threshold Value:", threshold_value)

# Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Color Image")
plt.imshow(color_img_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Histogram (Grayscale)")
plt.hist(gray_img.ravel(), 256)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 3, 3)
plt.title("Otsu Binarized Image")
plt.imshow(binary_img, cmap='gray')
plt.axis("off")

plt.show()
