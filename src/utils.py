import random
import numpy as np
import torch

SEED = 42

CLASS_MAP = {
    "car": 0,
    "person": 1,
    "truck": 2,
    "bus": 3,
    "motor": 4,
    "bike": 5,
    "traffic light": 6,
    "traffic sign": 7,
}

SELECTED_CLASSES = set(CLASS_MAP.keys())

FRCNN_CLASS_MAP = {cat: idx + 1 for cat, idx in CLASS_MAP.items()}

NUM_CLASSES = len(CLASS_MAP) + 1

CLASS_NAMES = {v: k for k, v in CLASS_MAP.items()}

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 255),
    (255, 128, 0),
]


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_environment():
    import ultralytics

    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("No GPU available")


def has_relevant_label(item):
    return any(
        obj.get("category") in SELECTED_CLASSES
        for obj in item.get("labels", [])
    )
