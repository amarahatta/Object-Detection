# Object Detection for Autonomous Vehicles

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0-red.svg)](https://pytorch.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.2.0-purple.svg)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview
 
This project benchmarks three object detection approaches — a modern one-stage detector (YOLOv8), a classical two-stage detector (Faster R-CNN) on the BDD100K autonomous driving dataset.
 
The goal is to quantitatively compare these methods across accuracy, speed, and model complexity, and to demonstrate why deep learning methods have become the standard for real-world autonomous vehicle perception systems.
 
**Key contributions:**
- Reproducible training and evaluation pipeline for all models on a stratified BDD100K subset
- Unified evaluation framework with mAP@0.5, mAP@0.5:0.95, per-class AP, FPS, and model size
- Targeted improvements addressing class imbalance and small object detection (resolution scaling, copy-paste augmentation, repeat factor sampling)
- Side-by-side qualitative analysis including failure case documentation

> **Demo:** https://huggingface.co/spaces/amarahatta/bdd-object-detection
> **Video:** https://youtu.be/CNPKmXZdYqc

## Features

- Preprocessed a 10,000-image BDD100K subset into train, validation, and test splits
- Converted BDD100K labels into YOLO-compatible annotation format
- Trained and compared YOLOv8m and Faster R-CNN ResNet-50 FPN
- Tested different training strategies including augmentation, higher input resolution, and repeat-factor sampling
- Evaluated models using mAP, precision, recall, F1 score, per-class AP, confusion matrices, FPS, and latency
- Built reusable helper modules for dataset loading, preprocessing, label conversion, sampling, and evaluation

## Project Structure

```text
RoadAware/
├── configs/
│   ├── yolov8_bdd100k.yaml
│   └── yolov8_bdd100k_v2.yaml
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_yolov8_default.ipynb
│   ├── 03_train_yolov8_augmentation.ipynb
│   ├── 03_train_yolov8_oversampling.ipynb
│   ├── 03_train_yolov8_resolution.ipynb
│   ├── 04_train_fasterrcnn_default.ipynb
│   ├── 04_train_fasterrcnn_augmentation.ipynb
│   ├── 04_train_fasterrcnn_oversampling.ipynb
│   ├── 04_train_fasterrcnn_resolution.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_train_yolov8_final.ipynb
│   ├── 07_train_fasterrcnn_final.ipynb
│   └── 08_evaluation_v2.ipynb
└── src/
    ├── bdd100k_io.py
    ├── convert_labels.py
    ├── evaluate.py
    ├── fasterrcnn_dataset.py
    ├── fasterrcnn_utils.py
    ├── preprocess.py
    ├── repeat_factor_sampler.py
    └── utils.py
```
## Classes

The model detects 8 traffic-related classes:

| ID | Class |
|---:|---|
| 0 | car |
| 1 | person |
| 2 | truck |
| 3 | bus |
| 4 | motor |
| 5 | bike |
| 6 | traffic light |
| 7 | traffic sign |

> **Note on class indexing:** Faster R-CNN reserves class 0 for background — all class IDs are shifted by +1 relative to YOLOv8. This is handled automatically in `src/fasterrcnn_dataset.py`.

## Dataset 
**Source:** [100k Labeled Road Images by SoleSensei](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k) (Kaggle)  
**Original dataset:** [Berkeley DeepDrive BDD100K](https://bdd-data.berkeley.edu/)

Expected dataset inputs:
```text
bdd100k/
├── images/
│   ├── 100k/train/
│   └── 100k/val/
└── labels/
    ├── det_20/det_train.json
    └── det_20/det_val.json
```

### Subset Used
 
The full BDD100K dataset contains 100k images. We use a stratified subset that preserves diversity across conditions:
 
| Split | Size | Stratification |
|---|---|---|
| Train | ~7,000 images | Weather × time-of-day |
| Val | ~1,500 images | Weather × time-of-day |
| Test | ~1,500 images | Held out — not used during training or tuning |
 
Stratification variables: weather (clear, rainy, foggy, snowy) × time of day (daytime, night, dawn/dusk).
 
**Processed dataset** (YOLO format, labels converted, blurry images removed):  

## Model Experiments

### YOLOv8

Experiments included:

- Default YOLOv8m fine-tuning
- Data augmentation
- Oversampling / class-imbalance handling
- Higher-resolution training
- Final YOLOv8 training configuration

### Faster R-CNN

Experiments included:

- Faster R-CNN ResNet-50 FPN baseline
- Data augmentation
- Repeat-factor sampling for class imbalance
- Higher-resolution training
- Final Faster R-CNN training configuration

## Results

Best YOLOv8m result:

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | FPS | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8m Resolution / v2 | 0.5151 | 0.2888 | 0.7576 | 0.7121 | 0.7342 | 64.9 | 15.4 ms/image |

Best Faster R-CNN result:

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 | FPS | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN Oversampling | 0.4589 | 0.2408 | 0.6499 | 0.6975 | 0.6728 | 34.9 | 28.7 ms/image |

YOLOv8m performed better overall, especially in inference speed and F1 score, making it more practical for near-real-time road-scene object detection.

## Setup

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
pip install torch torchvision ultralytics opencv-python numpy pandas matplotlib scikit-learn torchmetrics tqdm jupyter
```

## Usage

### 1. Run dataset exploration

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Preprocess BDD100K labels and images

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

This step creates YOLO-style labels and train/validation/test splits.

### 3. Train YOLOv8 models

```bash
jupyter notebook notebooks/03_train_yolov8_default.ipynb
jupyter notebook notebooks/03_train_yolov8_augmentation.ipynb
jupyter notebook notebooks/03_train_yolov8_oversampling.ipynb
jupyter notebook notebooks/03_train_yolov8_resolution.ipynb
```

### 4. Train Faster R-CNN models

```bash
jupyter notebook notebooks/04_train_fasterrcnn_default.ipynb
jupyter notebook notebooks/04_train_fasterrcnn_augmentation.ipynb
jupyter notebook notebooks/04_train_fasterrcnn_oversampling.ipynb
jupyter notebook notebooks/04_train_fasterrcnn_resolution.ipynb
```

### 5. Evaluate models

```bash
jupyter notebook notebooks/05_evaluation.ipynb
```

## Notes

- Update the dataset paths inside `configs/yolov8_bdd100k.yaml` before training YOLOv8.
- Trained model weights and the BDD100K dataset should not be committed to GitHub because of file size limits.
- Use `.gitignore` to exclude large files such as datasets, model checkpoints, `.pt` files, cache files, and notebook checkpoints.
