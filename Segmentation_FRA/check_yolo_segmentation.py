#!/usr/bin/env python3
import cv2
import os
import numpy as np

# Cartelle immagini e label
IMG_DIR = "/home/artswarms/Desktop/Segmentation_FRA/seg_images_jpg"
LABEL_DIR = "/home/artswarms/Desktop/Segmentation_FRA/20221219_acq002_labels_yolo_seg_multi"

# Dimensioni immagini per denormalizzazione
IMG_W = 3840
IMG_H = 2400

# Numero di immagini da controllare
NUM_CHECK = 10

# Elenca immagini
images = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")])[:NUM_CHECK]

# Avvia thread finestra per ridurre warning Qt
cv2.startWindowThread()

# Funzione per cercare ricorsivamente i file label
def find_label_file(label_name, base_dir):
    for root, dirs, files in os.walk(base_dir):
        if label_name in files:
            return os.path.join(root, label_name)
    return None

for img_name in images:
    img_path = os.path.join(IMG_DIR, img_name)
    
    # Costruzione corretta del nome del label
    label_name = os.path.splitext(img_name)[0].replace("_Img_", "_") + ".txt"
    
    # Cerca il file label ricorsivamente
    label_path = find_label_file(label_name, LABEL_DIR)
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Immagine non trovata: {img_name}")
        continue

    if label_path and os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line_num, line in enumerate(f, start=1):
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                
                class_id = parts[0]  # primo valore corrisponde a class_id
                coords = list(map(float, parts[1:]))
                pts = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i]*IMG_W)
                    y = int(coords[i+1]*IMG_H)
                    pts.append([x,y])
                pts = np.array(pts, np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
    else:
        print(f"⚠️ Label mancante: {label_name}")

    # Mostra immagine ridimensionata
    cv2.imshow("Segmentation check", cv2.resize(img, (960,600)))
    key = cv2.waitKey(0)
    if key == 27:  # ESC per uscire
        break

cv2.destroyAllWindows()
