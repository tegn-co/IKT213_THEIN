import cv2
import numpy as np
import matplotlib.pyplot as plt

print("OpenCV version:", cv2.__version__)

# Test: lag et enkelt bilde og vis det
img = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(img, (100, 100), 50, 255, -1)

plt.imshow(img, cmap='gray')
plt.title("Test image")
plt.show()
