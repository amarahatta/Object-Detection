import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from typing import List, Optional


def build_repeat_factor_sampler(
    dataset,
    num_classes: int,
    repeat_thresh: float = 0.01,
    verbose: bool = True
) -> WeightedRandomSampler:
    num_images = len(dataset)
    class_freq = np.zeros(num_classes + 1)

    if verbose:
        print("Computing class frequencies for repeat factor sampler...")

    for idx in range(num_images):
        try:
            _, target = dataset[idx]
        except Exception:
            continue

        if isinstance(target, dict) and "labels" in target:
            labels = target["labels"]
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            elif isinstance(labels, np.ndarray):
                pass
            else:
                continue

            for label in np.unique(labels):
                if 0 <= label <= num_classes:
                    class_freq[label] += 1

    class_freq = class_freq / num_images

    rep_factors_per_class = np.ones(num_classes + 1)
    for c in range(1, num_classes + 1):
        if class_freq[c] > 0:
            rep_factors_per_class[c] = np.sqrt(
                repeat_thresh / min(class_freq[c], repeat_thresh)
            )

    image_weights = []
    for idx in range(num_images):
        try:
            _, target = dataset[idx]
        except Exception:
            image_weights.append(1.0)
            continue

        if isinstance(target, dict) and "labels" in target:
            labels = target["labels"]
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            elif not isinstance(labels, np.ndarray):
                image_weights.append(1.0)
                continue

            if len(labels) == 0:
                image_weights.append(1.0)
            else:
                image_weights.append(float(np.max(rep_factors_per_class[labels])))
        else:
            image_weights.append(1.0)

    class_names = ["background", "car", "person", "truck", "bus", "motor", "bike", "traffic light", "traffic sign"]
    if verbose:
        print("\nClass sampling multipliers:")
        for c in range(1, num_classes + 1):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            print(f"  {name}: freq={class_freq[c]:.4f} -> multiplier={rep_factors_per_class[c]:.2f}x")

    sampler = WeightedRandomSampler(
        weights=image_weights,
        num_samples=num_images,
        replacement=True,
    )

    return sampler


def get_class_frequencies(dataset, num_classes: int) -> np.ndarray:
    num_images = len(dataset)
    class_freq = np.zeros(num_classes + 1)

    for idx in range(num_images):
        try:
            _, target = dataset[idx]
        except Exception:
            continue

        if isinstance(target, dict) and "labels" in target:
            labels = target["labels"]
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            elif isinstance(labels, np.ndarray):
                pass
            else:
                continue

            for label in np.unique(labels):
                if 0 <= label <= num_classes:
                    class_freq[label] += 1

    return class_freq / num_images