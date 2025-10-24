ABSTRACT

Monitoring bird populations in natural habitats is fundamental for ecological research and wildlife conservation. Traditional observation methods rely heavily on manual inspection of large volumes of imagery, which is both time-consuming and prone to observer bias, especially when birds appear small or partially occluded in high-resolution scenes. In this work, we present a deep learning–based pipeline for automated bird detection and segmentation in ultra-high-resolution images. Our method leverages a YOLOv8 model trained on a custom dataset of wild birds, combined with a tiling/slicing strategy specifically designed to preserve small-object visibility in large-format images. The dataset was split using an 80/10/10 train/validation/test ratio, and custom hyperparameters were introduced to improve convergence on highly imbalanced scenes.
To handle the resolution constraints typical of field imagery, we adopt a windowing approach that crops images into overlapping tiles before inference, while preserving and remapping original bounding boxes and masks for accurate training supervision. Experimental results demonstrate promising detection and segmentation performance, showing the viability of YOLOv8 for fine-grained bird localization under real-world environmental conditions. The proposed workflow automates a previously manual process, enabling scalable and efficient bird population monitoring and supporting long-term ecological observation efforts.


# Bird Detection via YOLOv8
*Francesca Grucci*

This repository provides the implementation of a bird detection pipeline using **YOLOv8**. The project focuses on detecting birds in images, using a custom dataset of bird images and labels. The pipeline includes dataset preparation, training, hyperparameter tuning, evaluation, and metric visualization. The approach leverages pretrained YOLOv8 weights to improve convergence and accuracy.

## Dataset

The dataset is organized in a standard YOLO format with images and corresponding labels:

- **Train/Val/Test split:** 80% / 10% / 10%
- **Counts:**  
  - Train: 102060 images  
  - Validation: 12757 images  
  - Test: 12758 images  
- **Classes:** `["bird"]`

The dataset YAML (`dataset.yaml`) is automatically created and contains paths to images, number of classes, and class names.

## Training

The model uses **YOLOv8n** as the base architecture, pretrained on COCO, with automatic optimizer selection (AdamW). Training is performed with:

- Image size: 1024  
- Batch size: 4  
- Device: GPU 0  
- Epochs: 2 (for test run, can be increased for final training)  
- Automatic Mixed Precision (AMP) enabled  
- Pretrained weights used for faster convergence

The training logs are saved in  /mnt/Atlante/SEG_YOLO_uccelli/dataset/uccello_exp


### Hyperparameters

Hyperparameters are defined in `hyp_birds.yaml`:

 ```yaml
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 7.5
cls: 0.5
dfl: 1.5 
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

The following metrics were obtained after a 2-epoch test run:

| Metric     | Value  |
|------------|--------|
| Precision  | 0.864  |
| Recall     | 0.800  |
| mAP50      | 0.858  |
| mAP50-95   | 0.473  |

##### Source
@misc{Grucci2025BirdYOLO,
  title={Bird Segmentation via YOLOv8},
  author={Grucci Francesca},
  year={2025},
  note={University project, custom dataset}
}
