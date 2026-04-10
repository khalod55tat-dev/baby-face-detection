[README_baby.md](https://github.com/user-attachments/files/26621539/README_baby.md)
# 👶 Baby Face Classifier

A side project I built from scratch to learn image detection and computer vision using PyTorch.

> Not the most accurate model out there, but I built it myself and learned a ton doing it. 🤷

---

## 🧠 What It Does

Detects whether a face in an image belongs to a **baby** or a **non-baby** using a custom-trained CNN model. It uses OpenCV for face detection and PyTorch for classification.

---

## 📁 Project Structure

```
baby-face-project/
├── data/faces/
│   ├── baby/          # Training images of babies
│   └── non_baby/      # Training images of non-babies
├── models/            # Saved model checkpoints (.pth)
├── step2_image_basics.py       # Image loading & basics
├── step3_face_detection.py     # Face detection with OpenCV
├── step4_check_dataset.py      # Dataset validation
├── step5_check_crop.py         # Face cropping logic
├── step5_crop_all_faces.py     # Batch face cropping
├── train_step6_baby_classifier.py  # Model training
├── predict_step7_baby_classifier.py # Run predictions
├── live_baby_detector.py       # Live webcam detection
└── test_image.py               # Test on a single image
```

---

## ⚙️ How It Works

1. **Face Detection** — Uses OpenCV's Haar Cascade to find faces in images
2. **Cropping** — Crops detected faces to 128x128 pixels
3. **Training** — Trains a CNN on baby vs non-baby face images using PyTorch
4. **Prediction** — Classifies new faces as baby or not

### Model Config
```python
Input size:    128x128
Batch size:    16
Epochs:        5
Learning rate: 0.001
Device:        CUDA (if available) else CPU
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/khalod55tat-dev/baby-face-project.git
cd baby-face-project

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install torch torchvision opencv-python

# 4. Add your training images
# Put baby face images in:     data/faces/baby/
# Put non-baby face images in: data/faces/non_baby/

# 5. Train the model
python train_step6_baby_classifier.py

# 6. Test on an image
python predict_step7_baby_classifier.py

# 7. Try live webcam detection
python live_baby_detector.py
```

---

## 📊 Results

Honestly? It works... sometimes. Built this to learn, not to ship. The model is decent with clear frontal faces but struggles with:
- Side profiles
- Low lighting
- Small faces

PRs welcome if you want to improve it 😄

---

## 📚 What I Learned

- How CNNs work under the hood
- PyTorch DataLoader and training loops
- OpenCV face detection with Haar Cascades
- The pain of collecting and cleaning training data

---

## 🛠️ Built With

- Python
- PyTorch & TorchVision
- OpenCV
- Haar Cascade (face detection)
