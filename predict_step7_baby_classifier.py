import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

# ========= 1. Config =========
MODEL_PATH = "models/baby_classifier_step6.pth"
CASCADE_PATH = "models/haarcascade_frontalface_default.xml"
IMAGE_PATH = "data/test.jpg"   # change this if you want

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========= 2. Model definition (same as in training) =========
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64x64

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16x16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ========= 3. Load model =========
model = SimpleCNN(num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded.")

# Class names MUST match training order from ImageFolder
class_names = ["baby", "non_baby"]

# ========= 4. Transforms =========
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ========= 5. Load image & detect face =========
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError("Could not load Haar cascade. Check CASCADE_PATH.")

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    img_gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(40, 40)
)

if len(faces) == 0:
    raise RuntimeError("No face detected in the image!")

# For simplicity, just use the first detected face
(x, y, w, h) = faces[0]
face_crop = img_bgr[y:y+h, x:x+w]

# Optional: draw box and show/save it
img_with_box = img_bgr.copy()
cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imwrite("data/test_with_box.jpg", img_with_box)
cv2.imwrite("data/test_face_crop.jpg", face_crop)

# Convert BGR -> RGB for transform
face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

# Convert to PIL-like tensor using transforms
# (transforms can accept numpy arrays via ToTensor, but we'll cleanly convert)
from PIL import Image
face_pil = Image.fromarray(face_rgb)

input_tensor = transform(face_pil)         # shape: (3, 128, 128)
input_batch = input_tensor.unsqueeze(0)    # shape: (1, 3, 128, 128)
input_batch = input_batch.to(device)

# ========= 6. Inference =========
with torch.no_grad():
    outputs = model(input_batch)           # (1, 2)
    probs = torch.softmax(outputs, dim=1)
    confidence, pred_class = torch.max(probs, 1)

pred_idx = pred_class.item()
pred_label = class_names[pred_idx]
conf_percent = confidence.item() * 100

print(f"Prediction: {pred_label} ({conf_percent:.2f}% confidence)")