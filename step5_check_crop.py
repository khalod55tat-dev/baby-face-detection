import os

FACES_DIR = "data/faces"
classes = ["baby", "non_baby"]

for cls in classes:
    folder = os.path.join(FACES_DIR, cls)
    files = [f for f in os.listdir(folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Class '{cls}': {len(files)} cropped face images")