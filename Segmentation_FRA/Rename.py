import os

LABELS_DIR = "20221219_acq002_labels_yolo_seg_multi"

for root, dirs, files in os.walk(LABELS_DIR):
    for file in files:
        if file.endswith(".txt") and "_Img_" in file:
            old_path = os.path.join(root, file)
            # Rimuovo "_Img_"
            new_name = file.replace("_Img_", "_")
            new_path = os.path.join(root, new_name)
            os.rename(old_path, new_path)
            print(f"Rinominato: {old_path} -> {new_path}")
