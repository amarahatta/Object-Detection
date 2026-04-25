"""Load BDD100K annotations: merged `bdd100k_labels_images_*.json` or per-image JSONs under bdd100k_labels/100k/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

# Align with rare alternate strings in raw labels vs `CLASS_MAP` in `src.utils`.
_CATEGORY_ALIASES = {
    "motorcycle": "motor",
    "bicycle": "bike",
}


def _norm_category(cat: Any) -> Any:
    if not isinstance(cat, str):
        return cat
    return _CATEGORY_ALIASES.get(cat, cat)


def raw_per_image_json_to_item(data: Dict[str, Any]) -> Dict[str, Any]:
    """One BDD100K per-image detection JSON -> legacy item (`name`, `attributes`, `labels`)."""
    name = data.get("name", "")
    attrs = dict(data.get("attributes") or {})
    labels: List[Dict[str, Any]] = []
    for frame in data.get("frames") or []:
        for obj in frame.get("objects") or []:
            if "box2d" not in obj:
                continue
            labels.append(
                {
                    "category": _norm_category(obj.get("category")),
                    "box2d": obj["box2d"],
                }
            )
    return {"name": name, "attributes": attrs, "labels": labels}


def _load_split_per_image(label_dir: Path, desc: str) -> List[Dict[str, Any]]:
    paths = sorted(label_dir.glob("*.json"))
    out: List[Dict[str, Any]] = []
    for p in tqdm(paths, desc=desc, unit="file"):
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        out.append(raw_per_image_json_to_item(raw))
    return out


def load_bdd100k_train_val(
    dataset_dir: Path,
    *,
    merged_dir: Path | None = None,
    label_root: Path | None = None,
    image_root: Path | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, str]]:
    """
    Returns ``(train_annotations, val_annotations, meta)`` where ``meta`` has string paths:
    ``TRAIN_IMAGE_DIR``, ``VAL_IMAGE_DIR``, ``TRAIN_JSON``, ``VAL_JSON``, ``source``.
    """
    dataset_dir = Path(dataset_dir)
    merged_dir = Path(merged_dir) if merged_dir is not None else dataset_dir / "Bdd100k"
    label_root = Path(label_root) if label_root is not None else dataset_dir / "bdd100k_labels" / "100k"
    image_root = Path(image_root) if image_root is not None else dataset_dir / "bdd100k_images_100k" / "100k"

    train_json = merged_dir / "bdd100k_labels_images_train.json"
    val_json = merged_dir / "bdd100k_labels_images_val.json"

    img_train = image_root / "train"
    img_val = image_root / "val"
    if not img_train.is_dir() or not img_val.is_dir():
        raise FileNotFoundError(
            "Expected image folders:\n"
            f"  {img_train}\n"
            f"  {img_val}\n"
            "Set `bdd100k_images_100k/100k/train` and `.../val` (or pass a custom `image_root`)."
        )

    meta_base: Dict[str, str] = {
        "TRAIN_IMAGE_DIR": str(img_train),
        "VAL_IMAGE_DIR": str(img_val),
        "TRAIN_JSON": str(train_json),
        "VAL_JSON": str(val_json),
    }

    if train_json.is_file() and val_json.is_file():
        with open(train_json, encoding="utf-8") as f:
            train = json.load(f)
        with open(val_json, encoding="utf-8") as f:
            val = json.load(f)
        meta = {**meta_base, "source": "merged_json"}
        return train, val, meta

    train_lbl = label_root / "train"
    val_lbl = label_root / "val"
    if not train_lbl.is_dir() or not val_lbl.is_dir():
        raise FileNotFoundError(
            "No merged labels in:\n"
            f"  {train_json}\n"
            f"  {val_json}\n"
            "and no per-image label folders:\n"
            f"  {train_lbl}\n"
            f"  {val_lbl}\n"
            "Place merged files under Bdd100k/ or per-image JSONs under bdd100k_labels/100k/."
        )

    train = _load_split_per_image(train_lbl, "train labels")
    val = _load_split_per_image(val_lbl, "val labels")
    meta = {**meta_base, "source": "per_image_json"}
    return train, val, meta
