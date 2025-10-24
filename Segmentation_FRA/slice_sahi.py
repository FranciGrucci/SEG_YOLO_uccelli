import cv2
import numpy as np
from pathlib import Path
import os

# ----------- CONFIG -----------
IMG_DIR = Path("/mnt/Atlante/SEG_YOLO_uccelli/Segmentation_FRA/seg_images_jpg")
LABEL_DIR = Path("/mnt/Atlante/SEG_YOLO_uccelli/Segmentation_FRA/20221219_acq002_labels_yolo_seg_multi")

OUTPUT_IMG_DIR = Path("/mnt/Atlante/SEG_YOLO_uccelli/images")
OUTPUT_LABEL_DIR = Path("/mnt/Atlante/SEG_YOLO_uccelli/labels")
OUTPUT_PREV_DIR = Path("/mnt/Atlante/SEG_YOLO_uccelli/previews")

OUTPUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREV_DIR.mkdir(parents=True, exist_ok=True)

IMG_W, IMG_H = 3840, 2400
SLICE_W, SLICE_H = 800, 800
OVERLAP_RATIO = 0.2

# ----------- FUNZIONI -----------
def read_yolo_seg(label_path):
    annotations = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            class_id = parts[0]
            coords = [float(x) for x in parts[1:]]
            annotations.append({"class": class_id, "coords": coords})
    return annotations

def write_yolo_seg(label_path, annotations):
    with open(label_path, "w") as f:
        for ann in annotations:
            line = ann["class"] + " " + " ".join(f"{c:.6f}" for c in ann["coords"])
            f.write(line + "\n")

def find_label_file(label_name, base_dir):
    """Cerca ricorsivamente la label nelle sottocartelle cam1/cam2/cam3"""
    for root, dirs, files in os.walk(base_dir):
        if label_name in files:
            return Path(root) / label_name
    return None

def crop_and_update_annotations(annotations, x0, y0, slice_w, slice_h):
    new_annotations = []
    for ann in annotations:
        coords = np.array(ann["coords"]).reshape(-1,2)
        coords[:,0] *= IMG_W
        coords[:,1] *= IMG_H
        inside_mask = (coords[:,0]>=x0) & (coords[:,0]<x0+slice_w) & (coords[:,1]>=y0) & (coords[:,1]<y0+slice_h)
        if not inside_mask.any():
            continue
        coords[:,0] = np.clip(coords[:,0]-x0,0,slice_w)/slice_w
        coords[:,1] = np.clip(coords[:,1]-y0,0,slice_h)/slice_h
        new_annotations.append({"class": ann["class"], "coords": coords.flatten().tolist()})
    return new_annotations

def draw_polygons(img, annotations):
    for ann in annotations:
        pts = np.array(ann["coords"]).reshape(-1,2)
        pts[:,0] = pts[:,0]*img.shape[1]
        pts[:,1] = pts[:,1]*img.shape[0]
        pts = pts.astype(np.int32)
        cv2.polylines(img,[pts],isClosed=True,color=(0,255,0),thickness=2)
    return img

# ----------- MAIN LOOP -----------
for img_file in IMG_DIR.iterdir():
    if img_file.suffix.lower() != ".jpg":
        continue

    # Nome label atteso
    label_name = img_file.stem.replace("_Img_","_") + ".txt"
    label_file = find_label_file(label_name, LABEL_DIR)

    if not label_file:
        print(f"⚠️ Label mancante: {label_name}")
        continue
    else:
        print(f"✅ Label trovata: {label_file}")

    img = cv2.imread(str(img_file))
    if img is None:
        print(f"⚠️ Immagine non trovata: {img_file.name}")
        continue

    annotations = read_yolo_seg(label_file)

    step_w = int(SLICE_W * (1-OVERLAP_RATIO))
    step_h = int(SLICE_H * (1-OVERLAP_RATIO))

    for y in range(0, IMG_H, step_h):
        for x in range(0, IMG_W, step_w):
            slice_img = img[y:y+SLICE_H, x:x+SLICE_W]
            slice_annotations = crop_and_update_annotations(annotations, x, y, SLICE_W, SLICE_H)

            slice_name = f"{img_file.stem}_x{x}_y{y}.jpg"
            slice_img_path = OUTPUT_IMG_DIR / slice_name
            slice_label_path = OUTPUT_LABEL_DIR / slice_name.replace(".jpg",".txt")

            # Salva la patch anche se vuota, così vedi output
            cv2.imwrite(str(slice_img_path), slice_img)
            if slice_annotations:
                write_yolo_seg(slice_label_path, slice_annotations)
                preview_img = draw_polygons(slice_img.copy(), slice_annotations)
                cv2.imwrite(str(OUTPUT_PREV_DIR / slice_name), preview_img)

print("✅ Slicing completato, controlla le cartelle!")
