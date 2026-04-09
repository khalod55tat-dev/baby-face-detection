import os

base_dir = "data/raw"
classes = ["baby", "non_baby"]

for cls in classes:
    folder = os.path.join(base_dir, cls)
    if not os.path.exists(folder):
        print(f"Folder missing: {folder}")
        continue

    files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"Class '{cls}': {len(files)} images")