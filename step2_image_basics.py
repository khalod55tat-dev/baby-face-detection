import cv2
import numpy as np

# Load image
img = cv2.imread("data/sample.jpg")

# Check shape
print("Image shape:", img.shape)

# Access pixel (row 100, col 100)
pixel = img[100, 100]
print("Pixel at (100, 100):", pixel)

# Split channels
b, g, r = cv2.split(img)
print("Blue channel shape:", b.shape)
print("Green channel shape:", g.shape)
print("Red channel shape:", r.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Gray shape:", gray.shape)

cv2.imwrite("data/gray_sample.jpg", gray)
# Resize to 224x224 (common size for neural networks)
resized = cv2.resize(img, (224, 224))
cv2.imwrite("data/resized_sample.jpg", resized)
# Draw rectangle: (x1, y1) to (x2, y2)
img_rect = img.copy()
cv2.rectangle(img_rect, (50, 50), (300, 300), (0, 255, 0), 3)
cv2.imwrite("data/rect_sample.jpg", img_rect)
i = 0
for (x, y, w, h) in faces:
    face_crop = img[y:y+h, x:x+w]
    cv2.imwrite(f"data/face_{i}.jpg", face_crop)
    i += 1