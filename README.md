# RoadAware

RoadAware is a computer vision project for road-scene object detection using dashcam images from the BDD100K dataset. The project trains and evaluates YOLOv8 and Faster R-CNN models to detect common traffic objects such as cars, people, trucks, buses, motorcycles, bikes, traffic lights, and traffic signs.

**Demo:** https://huggingface.co/spaces/AadMa/bdd-object-detection

**Video:** https://youtu.be/CNPKmXZdYqc

## Overview

The goal of this project is to compare modern object detection approaches for autonomous-driving-style perception tasks. The pipeline includes dataset exploration, preprocessing, class filtering, model training, class-imbalance handling, and evaluation using object detection metrics.

## Features

- Preprocessed a 10,000-image BDD100K subset into train, validation, and test splits
- Converted BDD100K labels into YOLO-compatible annotation format
- Trained and compared YOLOv8m and Faster R-CNN ResNet-50 FPN
- Tested different training strategies including augmentation, higher input resolution, and repeat-factor sampling
- Evaluated models using mAP, precision, recall, F1 score, per-class AP, confusion matrices, FPS, and latency
- Built reusable helper modules for dataset loading, preprocessing, label conversion, sampling, and evaluation

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

## Tech Stack

- **Languages:** Python
- **ML / Deep Learning:** PyTorch, TorchVision, Ultralytics YOLOv8
- **Computer Vision:** OpenCV
- **Data / Analysis:** NumPy, pandas, Matplotlib
- **Evaluation:** TorchMetrics, confusion matrices, per-class AP, FPS/latency benchmarking
- **Environment:** Jupyter Notebooks

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

## Dataset

This project uses the BDD100K object detection dataset. The full dataset is not included in this repository because of its size.

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

The preprocessing notebook filters images to the selected traffic classes and creates a smaller working subset:

```text
Train: 7000 images
Validation: 1500 images
Test: 1500 images
Total: 10000 images
```

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

## Suggested `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
.venv/

# Jupyter
.ipynb_checkpoints/

# Data and outputs
data/
datasets/
outputs/
runs/

# Model weights
*.pt
*.pth
*.onnx

# OS files
.DS_Store
```

## Author

**Michael Khuri**  
Portfolio: https://michaelkhuri.com  
GitHub: https://github.com/Savant-sys
