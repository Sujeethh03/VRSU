import cv2
import matplotlib.pyplot as plt
import numpy as np # Import numpy for potential use with gradients


img= cv2.imread("/content/340cbeb8-23fa-4a58-89fd-c3a7a3debcbf.jpg",0)

#1. Sobel edge Detection

sobel_x=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobel_y=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobel_xy=cv2.magnitude(sobel_x,sobel_y)


#2 Canny Edge Detection
canny_edges=cv2.Canny(img,100,200)

#Display Results

plt.figure(figsize=(12,6)) # Adjust figsize for better display of 4 images

plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(img,cmap='gray')
plt.axis("off")


plt.subplot(2,2,2)
plt.title("Sobel XY Edge")
plt.imshow(sobel_xy,cmap='gray') # Display Sobel XY result
plt.axis("off")

plt.subplot(2,2,3)
plt.title("Canny Edges")
plt.imshow(canny_edges,cmap='gray') # Display Canny Edges result
plt.axis("off")

plt.tight_layout() # Adjust subplot params for a tight layout
plt.show()