import os
import json
import random
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils import SEED, SELECTED_CLASSES, has_relevant_label


def filter_relevant_images(annotations):
    filtered = [item for item in annotations if has_relevant_label(item)]
    dropped = len(annotations) - len(filtered)
    print(
        f"Kept {len(filtered)} / {len(annotations)} images "
        f"({len(filtered) / len(annotations) * 100:.1f}%) — "
        f"dropped {dropped} with no selected-class labels"
    )
    return filtered


def _get_strata(item):
    attrs = item.get("attributes", {})
    weather = attrs.get("weather", "unknown")
    timeofday = attrs.get("timeofday", "unknown")
    return f"{weather}_{timeofday}"


def stratified_subset(annotations, train_size=15000, val_size=3000, test_size=2000):
    random.seed(SEED)
    np.random.seed(SEED)

    strata = defaultdict(list)
    for item in annotations:
        key = _get_strata(item)
        strata[key].append(item)

    total = train_size + val_size + test_size
    strata_proportions = {}
    for key, items in strata.items():
        strata_proportions[key] = len(items) / len(annotations)

    train_items, val_items, test_items = [], [], []
    for key, items in strata.items():
        random.shuffle(items)
        n = max(1, int(round(strata_proportions[key] * total)))
        n_train = max(1, int(round(strata_proportions[key] * train_size)))
        n_val = max(1, int(round(strata_proportions[key] * val_size)))
        n_test = max(1, int(round(strata_proportions[key] * test_size)))

        n_available = len(items)
        n_train = min(n_train, n_available)
        remaining = n_available - n_train
        n_val = min(n_val, remaining)
        remaining -= n_val
        n_test = min(n_test, remaining)

        train_items.extend(items[:n_train])
        val_items.extend(items[n_train : n_train + n_val])
        test_items.extend(items[n_train + n_val : n_train + n_val + n_test])

    random.shuffle(train_items)
    random.shuffle(val_items)
    random.shuffle(test_items)

    train_items = train_items[:train_size]
    val_items = val_items[:val_size]
    test_items = test_items[:test_size]

    print(f"Subset sizes — Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)}")
    return train_items, val_items, test_items


def split_annotations(annotations, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    random.seed(SEED)
    np.random.seed(SEED)

    indices = list(range(len(annotations)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_ratio + test_ratio), random_state=SEED
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=relative_test, random_state=SEED
    )

    train = [annotations[i] for i in train_idx]
    val = [annotations[i] for i in val_idx]
    test = [annotations[i] for i in test_idx]

    print(f"Split — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def copy_subset_files(annotations, split_name, src_image_dir, dst_image_dir, dst_label_dir, label_src_dir=None):
    import shutil

    os.makedirs(os.path.join(dst_image_dir, split_name), exist_ok=True)
    os.makedirs(os.path.join(dst_label_dir, split_name), exist_ok=True)

    count = 0
    for item in annotations:
        img_name = item["name"]
        src_img = os.path.join(src_image_dir, img_name)
        dst_img = os.path.join(dst_image_dir, split_name, img_name)

        if not os.path.exists(src_img):
            continue

        shutil.copy2(src_img, dst_img)

        label_name = img_name.replace(".jpg", ".txt")
        if label_src_dir:
            src_label = os.path.join(label_src_dir, label_name)
            dst_label = os.path.join(dst_label_dir, split_name, label_name)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

        count += 1

    print(f"Copied {count} images to {split_name}")
    return count
