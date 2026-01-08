#!/usr/bin/env python3
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
IMG_DIR = Path("/mnt/Atlante/SEG_YOLO_uccelli/images")
LABEL_DIR = Path("/mnt/Atlante/SEG_YOLO_uccelli/labels")
DATASET_DIR = Path("/mnt/Atlante/SEG_YOLO_uccelli/dataset")
DATASET_DIR.mkdir(exist_ok=True)

# Cartelle YOLO
IMG_TRAIN_DIR = DATASET_DIR / "images/train"
IMG_VAL_DIR = DATASET_DIR / "images/val"
IMG_TEST_DIR = DATASET_DIR / "images/test"
LABEL_TRAIN_DIR = DATASET_DIR / "labels/train"
LABEL_VAL_DIR = DATASET_DIR / "labels/val"
LABEL_TEST_DIR = DATASET_DIR / "labels/test"

for p in [IMG_TRAIN_DIR, IMG_VAL_DIR, IMG_TEST_DIR, LABEL_TRAIN_DIR, LABEL_VAL_DIR, LABEL_TEST_DIR]:
    p.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["bird"]

# ---------------- SPLIT DATASET 80/10/10 ----------------
all_images = list(IMG_DIR.glob("*.png")) + list(IMG_DIR.glob("*.jpg"))



# crea lista con numero di uccelli per immagine
num_birds_per_img = []
for img in all_images:
    label_file = LABEL_DIR / img.name.replace(".png", ".txt")
    if label_file.exists():
        with open(label_file) as f:
            num_birds = len(f.readlines())  # ogni riga = un uccello
    else:
        num_birds = 0
    num_birds_per_img.append(num_birds)

# split train / temp (80/20) stratificando per numero di uccelli
train_imgs, temp_imgs, _, temp_labels = train_test_split(
    all_images, num_birds_per_img, 
    test_size=0.2, random_state=42, 
    stratify=num_birds_per_img
)

# split val / test (50/50 del 20%) stratificando
val_imgs, test_imgs, _, _ = train_test_split(
    temp_imgs, temp_labels,
    test_size=0.5, random_state=42,
    stratify=temp_labels
)

print(f"✅ Split completato: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

# ---------------- CREA SYMLINK INVECE DI COPIA ----------------


def link_images_and_labels(img_list, img_dest, label_dest):
    for img in img_list:
        # symlink immagine
        img_link = img_dest / img.name
        if not img_link.exists():
            os.symlink(img.resolve(), img_link)

        # symlink label (solo se esiste)
        label_file = LABEL_DIR / img.name.replace(".png",".txt")
        if label_file.exists():
            label_link = label_dest / label_file.name
            if not label_link.exists():
                os.symlink(label_file.resolve(), label_link)


link_images_and_labels(train_imgs, IMG_TRAIN_DIR, LABEL_TRAIN_DIR)
link_images_and_labels(val_imgs, IMG_VAL_DIR, LABEL_VAL_DIR)
link_images_and_labels(test_imgs, IMG_TEST_DIR, LABEL_TEST_DIR)

print(f"Train/Val/Test counts: {len(train_imgs)}/{len(val_imgs)}/{len(test_imgs)}")

# ---------------- CREA FILE YAML DATASET ----------------
dataset_yaml = {
    "train": str(IMG_TRAIN_DIR),
    "val": str(IMG_VAL_DIR),
    "test": str(IMG_TEST_DIR),
    "nc": len(CLASS_NAMES),
    "names": CLASS_NAMES
}

yaml_path = DATASET_DIR / "dataset.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(dataset_yaml, f)
print(f"✅ YAML dataset creato: {yaml_path}")

# ---------------- CREA FILE HYPERPARAMETERS YOLOv8 ---------------

hyp_params = {
    "lr0": 0.0005,
    "lrf": 0.1,
    "momentum": 0.937,
    "weight_decay": 0.0008,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 10,
    "cls": 0.9,
    "dfl": 1.5,

      # --- Geometric Augmentation ---
    "degrees": 15.0,           # 🔼 aumentato da 10.0 - rotazione
    "translate": 0.2,          # 🔼 aumentato da 0.1 - traslazione
    "perspective": 0.0002,     # ✨ AGGIUNTO - distorsione prospettica

     # --- Data Augmentation ---
    "hsv_h": 0.03,    # variazione tonalità
    "hsv_s": 0.7,      # saturazione
    "hsv_v": 0.4,      # luminosità
    
    "scale": 1.2,      # zoom
    "shear": 2.0,      # inclinazione
    "flipud": 0.0,     # flip verticale
    "fliplr": 0.0,     # flip orizzontale
    "mosaic": 0.0,     # combinazione immagini (mosaic)
    "mixup": 0.2,      # blending di immagini
    "copy_paste": 0.3, # augmentation per oggetti piccoli
     # --- Noise & Blur (opzionali ma utili) ---
     "erasing": 0.4,          # random erasing (simula occlusioni)
     "auto_augment": "randaugment",  # augmentation automatica
}

hyp_path = DATASET_DIR / "hyp_birds.yaml"
with open(hyp_path, "w") as f:
    yaml.dump(hyp_params, f)
print(f"✅ YAML hyperparam creato: {hyp_path}")

# ---------------- PA-+
# 
# +RAMETRI TRAINING MODIFICABILI ----------------
#ODEL_NAME = "yolov8m-seg.pt"
MODEL_NAME = "yolov9c-seg.pt"
BATCH_SIZE = 1
DEVICE = 0  # 0=GPU, -1=CPU
EPOCHS = 300
IMGSZ = 640
BATCH_SIZE = 16


# ---------------- AVVIO TRAINING YOLOv8 ----------------
print("🚀 Avvio training YOLOv8...")
model = YOLO(MODEL_NAME)
train_results = model.train(
    data=str(yaml_path),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH_SIZE,
    freeze='bn',
    workers=2,
    device=DEVICE,
    name="uccello_exp",
    project=str(DATASET_DIR),
    patience=50,
    max_det=3000,    # soglia IoU per NMS (default = 0.7)
    agnostic_nms=False
)

# ---------------- EVALUATION SUL TEST SET ----------------
print("🔍 Valutazione sul test set...")
metrics = model.val(data=str(yaml_path),
    split="test",
    conf=0.10,   # soglia di confidenza (default = 0.25)
    iou=0.3
)

# ---------------- SALVA GRAFICI METRICHE ----------------
plots_dir = DATASET_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

# Precision-Recall curve
if hasattr(metrics, 'pr_curve'):
    plt.figure(figsize=(8,6))
    plt.plot(metrics.pr_curve["r"], metrics.pr_curve["p"], label="Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(plots_dir / "precision_recall_curve.png")
    plt.close()

# mAP evolution per classe
if hasattr(metrics, 'data') and 'map50' in metrics.data:
    plt.figure(figsize=(8,6))
    plt.bar(CLASS_NAMES, metrics.data['map50'])
    plt.xlabel("Classi")
    plt.ylabel("mAP50")
    plt.title("mAP50 per Classe")
    plt.grid(True)
    plt.savefig(plots_dir / "mAP50_per_class.png")
    plt.close()

print(f"✅ Training e valutazione completati! Grafici metriche salvati in {plots_dir}")
