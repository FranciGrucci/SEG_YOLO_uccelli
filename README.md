ABSTRACT

In this project, I developed a deep learning pipeline for automatic bird detection and segmentation in high resolution images. I used a YOLOv8 model trained on a custom dataset of wild birds and applied a tiling strategy to preserve the visibility of small objects in large images. The dataset was split 80/10/10 for training, validation, and testing, and I tuned custom hyperparameters to improve convergence on imbalanced scenes.

To handle high resolution constraints, I integrated the SAHI (Slicing Aided Hyper Inference) framework, which crops images into overlapping tiles while preserving accurate bounding box and mask mappings. Model performance was evaluated using the Intersection over Union (IoU) metric, showing that YOLOv8 achieves strong detection and segmentation results under real world conditions. Overall, the proposed approach makes bird population monitoring more efficient and scalable


# Bird Detection and Segmentation with YOLOv8 and SAHI
*Francesca Grucci*

This repository contains the implementation of a bird detection and instance segmentation pipeline using **YOLOv8** and **SAHI (Slicing Aided Hyper Inference)** for improved performance on high-resolution images with small objects.

---



## Dataset

The dataset is organized in a standard YOLO format with images and corresponding labels:

- **Train/Val/Test split:** 80% / 10% / 10%
- **Counts:**  
  - Train: 102060 images  
  - Validation: 12757 images  
  - Test: 12758 images  
- **Classes:** `["bird"]`

The dataset YAML (`dataset.yaml`) is automatically created and contains paths to images, number of classes (in this case 1), and class names ('bird').

dataset/
├── images/
│ ├── train/
│ ├── val/
│ └── test/
├── labels/
│ ├── train/
│ ├── val/
│ └── test/
└── dataset.yaml

The `dataset.yaml` file contains the paths to train, val, and test images, along with the class names.

---

The YOLOv8n model was trained for **2 epochs** (for testing purposes; full training can be extended) with the following parameters:

- **Model:** `yolov8n.pt`
- **Image size:** 1024
- **Batch size:** 4
- **Device:** GPU 
- **Epochs:** 2 (for test run, MUST be increased for final training)
- **Hyperparameters:**
  - Learning rate (`lr0`): 0.001 and (`lrf`):  0.01
  - Momentum: 0.937
  - Weight decay: 0.0005
  - Box loss gain: 7.5
  - Class loss gain: 0.5
  - DFL loss gain: 1.5

Training logs and weights are stored in:

/mnt/Atlante/SEG_YOLO_uccelli/dataset/uccello_exp



---

## Segmentation and Slicing with SAHI

To handle high-resolution images and small birds, **SAHI** is used for slicing the images during inference:

- **Instance Segmentation:** YOLOv8 predicts both bounding boxes and segmentation masks.
- **Slicing:** Images are divided into overlapping tiles to detect small birds accurately.
- **Merging:** Predictions from all slices are merged to form full-image results.
- **Advantage:** Preserves fine details and improves detection and segmentation metrics without resizing the entire image.



---
### Segmentation Label Generation

The segmentation labels are created using a script that performs the following steps:

1. **Parse contours:**  
   Each camera has a `CAM*_countorns.dat` file containing coordinates of detected objects. These are read and stored in memory.

2. **Build polygons:**  
   - **Alphashape** is used to create concave polygons if installed.  
   - **Shapely** can generate convex hulls.  
   - A fallback **monotone chain convex hull algorithm** ensures a polygon is always created.

3. **Normalize coordinates:**  
   Polygon points are scaled to `[0,1]` relative to the image dimensions (`3840x2400`).

4. **Write YOLOv8 `.txt` label files:**  
   - Format: `<class_id> x1 y1 x2 y2 x3 y3 ... xn yn`  
   - Empty polygons are discarded.  
   - Atomic file writing ensures safety.

5. **Process multiple cameras:**  
   - Images are processed sequentially for CAM1, CAM2, CAM3.  
  

---

## SAHI Slicing

Large images can miss small objects if processed directly. SAHI improves inference as follows:

1. **Slice images:**  
   Images are divided into overlapping tiles.

2. **Run YOLOv8 inference per tile:**  
   Each slice is processed independently to detect small objects accurately.

3. **Merge results:**  
   Overlapping predictions are combined, and duplicates are removed using non-max suppression.  
   The final output is a set of polygons corresponding to the original image.

---

## Installation

To run this project, install the required packages:

```bash

pip install ultralytics sahi matplotlib

```

#### Evaluaion

After training, the model is evaluated on the test set


Precision: 0.864

Recall: 0.800

mAP50: 0.858

mAP50-95: 0.473


Metrics are saved and visualized automatically in /mnt/Atlante/SEG_YOLO_uccelli/dataset/plots:

Precision-Recall Curve

mAP50 per class

##### Usage

1. Clone the repository and prepare your dataset in YOLO format (images + .txt labels).

2. Adjust parameters in the script YoLo.py:

EPOCHS = 2
IMGSZ = 1024
BATCH_SIZE = 4
DEVICE = 0  # GPU 0, CPU=-1
MODEL_NAME = "yolov8n.pt"

3. Run

python YoLo.py

###### Results


The following metrics were obtained after a 2-epoch test run on the **test set**:

| Metric       | Value  |
|-------------|--------|
| Precision   | 0.864  |
| Recall      | 0.800  |
| mAP50       | 0.858  |
| mAP50-95    | 0.473  |

The model demonstrates good detection and segmentation performance even on small objects, thanks to the combination of YOLOv8 and SAHI slicing.

![Precision-Recall Curve](plots/precision_recall_curve.png)
![mAP50 per Class](plots/mAP50_per_class.png)


### 📊 Final Results (20 epochs)

Il modello è stato addestrato per 20 epoche su YOLOv8n con slicing SAHI su immagini 1024×1024.
Rispetto al test iniziale a 2 epoche, le metriche sono migliorate sensibilmente, indicando un training stabile e progressivo senza overfitting.

| Metric     | Value |
|-----------|--------|
| Precision | 0.927 |
| Recall    | 0.861 |
| mAP50     | 0.936 |
| mAP50-95  | 0.612 |

Questi valori confermano che il modello è in grado di generalizzare bene anche su immagini non viste.

---

## Pipeline Overview

High-level workflow of the bird segmentation and detection pipeline:

Images (High-Res) 
       │
       ▼
Segmentation Label Generation
       │
       ├─ Parse contours (`CAM*_countorns.dat`)
       ├─ Build polygons (Alphashape / Shapely / Convex Hull)
       └─ Normalize coordinates & write YOLOv8 `.txt` labels
       │
       ▼
SAHI Slicing (Optional for inference)
       │
       ├─ Slice large images into tiles
       ├─ Run YOLOv8 inference per tile
       └─ Merge overlapping predictions
       │
       ▼
YOLOv8 Training (Segmentation)
       │
       ├─ Pretrained model: `yolov8n.pt`
       ├─ Dataset: Train / Val / Test split
       ├─ Hyperparameters: AdamW, batch=4, img=1024
       └─ Output: Model weights & metrics
       │
       ▼
Evaluation & Results
       │
       └─ Precision / Recall / mAP metrics





##### Source
@misc{Grucci2025BirdYOLO,
  title={Bird Segmentation via YOLOv8},
  author={Grucci Francesca},
  year={2025},
  note={University project, custom dataset}
}



