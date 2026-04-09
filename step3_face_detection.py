import cv2
import matplotlib.pyplot as plt

# Load the face detection model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load image
img = cv2.imread("data/sample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(50, 50)
)

print("Number of faces detected:", len(faces))

# Draw boxes
img_with_boxes = img.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)

# Convert to RGB for displaying
img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis("off")
plt.show()
i = 0
for (x, y, w, h) in faces:
    face_crop = img[y:y+h, x:x+w]
    cv2.imwrite(f"data/face_{i}.jpg", face_crop)
    i += 1
    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=3,
    minSize=(20, 20)
)