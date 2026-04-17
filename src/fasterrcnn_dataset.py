import os
import json

import torch
from torch.utils.data import Dataset
import cv2


class BDD100KDataset(Dataset):
    def __init__(self, image_dir, annotation_file, class_map, transforms=None):
        self.image_dir = image_dir
        self.class_map = class_map
        self.transforms = transforms

        with open(annotation_file) as f:
            all_annotations = json.load(f)

        self.annotations = [
            item for item in all_annotations
            if any(obj.get("category") in class_map
                   for obj in item.get("labels", []))
        ]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        img_path = os.path.join(self.image_dir, item["name"])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        boxes, labels = [], []
        for obj in item.get("labels", []):
            category = obj.get("category")
            if category not in self.class_map:
                continue
            box = obj["box2d"]
            x1, y1 = max(0, box["x1"]), max(0, box["y1"])
            x2, y2 = min(img_w, box["x2"]), min(img_h, box["y2"])

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_map[category])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        if self.transforms:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target
