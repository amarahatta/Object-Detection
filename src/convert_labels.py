import os
import json
import warnings

import cv2
import matplotlib.pyplot as plt

from src.utils import CLASS_MAP, CLASS_NAMES, COLORS

def _normalize_bdd100k_name(name):
    """Strip whitespace; use basename if ``name`` looks like a relative path (merged JSON style)."""
    if not isinstance(name, str):
        name = str(name) if name is not None else ""
    name = name.strip()
    if not name:
        return name
    if "/" in name or "\\" in name:
        name = os.path.basename(name.replace("\\", "/"))
    return name


def _find_bdd100k_image(img_filename, image_dirs):
    """Resolve disk path; BDD `name` is often a stem without extension (images are ``stem.jpg``)."""
    img_filename = _normalize_bdd100k_name(img_filename)
    if not img_filename:
        return None
    for root in image_dirs:
        candidate = os.path.join(root, img_filename)
        if os.path.isfile(candidate):
            return candidate
    stem, ext = os.path.splitext(img_filename)
    if ext.lower() not in (".jpg", ".jpeg", ".png"):
        for ext in (".jpg", ".jpeg", ".png"):
            for root in image_dirs:
                candidate = os.path.join(root, stem + ext)
                if os.path.isfile(candidate):
                    return candidate
    return None


def _validate_image_roots(image_dirs):
    roots = [os.path.abspath(os.path.expanduser(d)) for d in image_dirs]
    missing = [r for r in roots if not os.path.isdir(r)]
    if missing:
        raise FileNotFoundError(
            "BDD100K image folder(s) not found (check DATASET_PATH / TRAIN_IMAGE_DIR / VAL_IMAGE_DIR):\n"
            + "\n".join(f"  {m}" for m in missing)
        )
    for r in roots:
        found = False
        with os.scandir(r) as it:
            for e in it:
                if e.is_file() and e.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    found = True
                    break
        if not found:
            warnings.warn(f"No .jpg/.jpeg/.png files found under {r!r}.", stacklevel=2)
    return roots


def _yolo_label_basename(img_filename):
    """``stem.txt`` for YOLO labels (strip image extension if present)."""
    stem, ext = os.path.splitext(img_filename)
    if ext.lower() in (".jpg", ".jpeg", ".png"):
        return stem + ".txt"
    return img_filename + ".txt"


def convert_bdd100k_to_yolo(json_path, image_dirs, output_dir, *, miss_log_max=10):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        annotations = json.load(f)

    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]
    image_dirs = _validate_image_roots(image_dirs)

    converted = 0
    skipped_no_img = 0
    skipped_no_labels = 0
    miss_logged = 0

    for item in annotations:
        img_filename = _normalize_bdd100k_name(item.get("name", ""))

        img_path = _find_bdd100k_image(img_filename, image_dirs)

        if img_path is None:
            skipped_no_img += 1
            if miss_log_max is None or miss_logged < miss_log_max:
                print(f"[MISS] {img_filename}")
                miss_logged += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            skipped_no_img += 1
            print(f"[BAD IMAGE] {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        lines = []
        for obj in item.get("labels", []):
            category = obj.get("category")
            if category not in CLASS_MAP:
                continue
            if "box2d" not in obj:
                continue

            box = obj["box2d"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            lines.append(f"{CLASS_MAP[category]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not lines:
            skipped_no_labels += 1
            continue

        out_name = _yolo_label_basename(img_filename)
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, "w") as f:
            f.write("\n".join(lines))

        converted += 1

    print(f"Converted: {converted}")
    print(f"Skipped (no image file): {skipped_no_img}")
    if skipped_no_img and miss_log_max is not None and skipped_no_img > miss_logged:
        print(
            f"  (only first {miss_logged} [MISS] lines printed; "
            "pass miss_log_max=None to print every miss)"
        )
    print(f"Skipped (no selected-class labels): {skipped_no_labels}")
    return converted


def draw_bdd100k_labels(image_path, labels):
    """Draw legacy-format ``labels`` (``category`` + ``box2d``) on an image; only ``CLASS_MAP`` classes."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for obj in labels:
        category = obj.get("category")
        if category not in CLASS_MAP:
            continue
        if "box2d" not in obj:
            continue
        box = obj["box2d"]
        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])
        cls = CLASS_MAP[category]
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            CLASS_NAMES[cls],
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return img


def draw_yolo_labels(image_path, label_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        return img

    with open(label_path) as f:
        for line in f.read().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = map(float, parts)
            cls = int(cls)
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            color = COLORS[cls % len(COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, CLASS_NAMES[cls], (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )
    return img


def visual_verification(image_paths, label_dir, output_path, grid_size=3, num_samples=9):
    import random

    samples = random.sample(image_paths, min(num_samples, len(image_paths)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 12))
    for ax, img_path in zip(axes.flat, samples):
        label_path = os.path.join(
            label_dir,
            os.path.basename(img_path).replace(".jpg", ".txt"),
        )
        annotated = draw_yolo_labels(img_path, label_path)
        if annotated is not None:
            ax.imshow(annotated)
        ax.axis("off")

    plt.suptitle("Post-Conversion Verification — Random Sample", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Verification grid saved to {output_path}")
