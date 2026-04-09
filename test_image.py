import cv2
import matplotlib.pyplot as plt

# 1. Read image (BGR format by default)
img_bgr = cv2.imread("data/sample.jpg")

if img_bgr is None:
    raise FileNotFoundError("Image not found. Check the path and file name!")

# 2. Convert BGR -> RGB (so colors display correctly)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 3. Show image
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Sample Image")
plt.show()