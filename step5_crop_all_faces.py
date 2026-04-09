import os
import cv2

# Paths
RAW_DIR = "data/raw"
FACES_DIR = "data/faces"
CASCADE_PATH = "models/haarcascade_frontalface_default.xml"

# Load face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    raise RuntimeError("Could not load Haar cascade. Check the path to haarcascade_frontalface_default.xml")

# Classes we have
classes = ["baby", "non_baby"]

# Make sure output folders exist
for cls in classes:
    out_dir = os.path.join(FACES_DIR, cls)
    os.makedirs(out_dir, exist_ok=True)

for cls in classes:
    raw_folder = os.path.join(RAW_DIR, cls)
    out_folder = os.path.join(FACES_DIR, cls)

    if not os.path.exists(raw_folder):
        print(f"[WARN] Folder does not exist: {raw_folder}")
        continue

    files = [f for f in os.listdir(raw_folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"\nProcessing class '{cls}' with {len(files)} images...")

    img_count = 0
    face_count = 0

    for fname in files:
        img_path = os.path.join(raw_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIP] Could not read image: {img_path}")
            continue

        img_count += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        if len(faces) == 0:
            print(f"[NO FACE] {img_path}")
            continue

        # Save each detected face
        for i, (x, y, w, h) in enumerate(faces):
            face_crop = img[y:y+h, x:x+w]

            # Optional: resize to a fixed size, e.g. 128x128
            face_crop = cv2.resize(face_crop, (128, 128))

            out_name = f"{os.path.splitext(fname)[0]}_face{i}.jpg"
            out_path = os.path.join(out_folder, out_name)
            cv2.imwrite(out_path, face_crop)
            face_count += 1

    print(f"Done class '{cls}': {img_count} images processed, {face_count} faces saved.")