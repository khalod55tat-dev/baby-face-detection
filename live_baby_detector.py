import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ======= 1. Load Model =======
MODEL_PATH = "models/baby_classifier_step6.pth"
CASCADE_PATH = "models/haarcascade_frontalface_default.xml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# SAME MODEL AS TRAINING
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
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

model = SimpleCNN(num_classes=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded.")

# Class names
class_names = ["baby", "non_baby"]

# ======= 2. Preprocessing =======
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ======= 3. Load Face Detector =======
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ======= 4. Start Webcam =======
cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    raise RuntimeError("Could not access webcam.")

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        # Draw box
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)

        # Crop
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Prepare for model
        face_pil = Image.fromarray(face_rgb)
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(face_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_class = torch.max(probs, 1)

        label = class_names[pred_class.item()]
        conf = float(confidence.item()) * 100

        # ======= 5. Draw result text =======
        if label == "baby":
            color = (0, 255, 0)  # green
            text = f"BABY ({conf:.1f}%)"
        else:
            color = (0, 0, 255)  # red
            text = f"NON-BABY ({conf:.1f}%)"

        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show live video
    cv2.imshow("Baby Classifier", frame)

    # Exit when pressing "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()