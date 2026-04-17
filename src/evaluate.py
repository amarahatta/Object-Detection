import os
import time

import numpy as np
import torch
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def predict_yolov8(model, image_path, conf=0.25, iou=0.45):
    results = model.predict(image_path, conf=conf, iou=iou, verbose=False)
    preds = []
    for r in results:
        for box in r.boxes:
            preds.append({
                "box": box.xyxy[0].cpu().numpy().tolist(),
                "score": float(box.conf[0]),
                "label": int(box.cls[0]) + 1,
            })
    return preds


def predict_fasterrcnn(model, image_tensor, device, score_thresh=0.5):
    model.eval()
    with torch.no_grad():
        outputs = model([image_tensor.to(device)])[0]
    preds = []
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        if score >= score_thresh:
            preds.append({
                "box": box.cpu().numpy().tolist(),
                "score": float(score),
                "label": int(label),
            })
    return preds


def compute_map(predictions, targets):
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(predictions, targets)
    result = metric.compute()
    return result


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


def compute_precision_recall_f1(predictions, targets, iou_threshold=0.5):
    TP = FP = FN = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        matched_gt = set()
        for pb, pl in zip(pred_boxes, pred_labels):
            best_iou, best_idx = 0, -1
            for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if pl != gl or i in matched_gt:
                    continue
                pb_np = pb.cpu().numpy() if isinstance(pb, torch.Tensor) else pb
                gb_np = gb.cpu().numpy() if isinstance(gb, torch.Tensor) else gb
                iou = compute_iou(pb_np, gb_np)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= iou_threshold:
                TP += 1
                matched_gt.add(best_idx)
            else:
                FP += 1
        FN += len(gt_boxes) - len(matched_gt)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1


def measure_fps(model_predict_fn, test_images, n_warmup=10):
    for img in test_images[:n_warmup]:
        _ = model_predict_fn(img)

    start = time.perf_counter()
    for img in test_images:
        _ = model_predict_fn(img)
    elapsed = time.perf_counter() - start

    fps = len(test_images) / elapsed
    ms_per_image = (elapsed / len(test_images)) * 1000
    return fps, ms_per_image


def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def count_all_params(model):
    return sum(p.numel() for p in model.parameters())

def plot_pr_curve(precisions, recalls, model_name, save_path):
    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, marker=".", label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve — {model_name}")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PR curve saved to {save_path}")


def build_confusion_matrix(predictions, targets, class_names, iou_threshold=0.5, model_name="", save_path=None):
    n = len(class_names) + 1
    cm = np.zeros((n, n), dtype=int)
    bg_idx = len(class_names)

    label_to_idx = {}
    for i, name in enumerate(class_names):
        label_to_idx[i + 1] = i

    for pred, target in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        matched_gt = set()
        for pb, pl in zip(pred_boxes, pred_labels):
            best_iou, best_idx = 0, -1
            for i, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if i in matched_gt:
                    continue
                pb_np = pb.cpu().numpy() if isinstance(pb, torch.Tensor) else pb
                gb_np = gb.cpu().numpy() if isinstance(gb, torch.Tensor) else gb
                iou = compute_iou(pb_np, gb_np)
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            pred_idx = label_to_idx.get(int(pl), bg_idx)
            if best_iou >= iou_threshold and best_idx >= 0:
                gt_idx = label_to_idx.get(int(gt_labels[best_idx]), bg_idx)
                cm[gt_idx, pred_idx] += 1
                matched_gt.add(best_idx)
            else:
                cm[bg_idx, pred_idx] += 1

        for i, gl in enumerate(gt_labels):
            if i not in matched_gt:
                gt_idx = label_to_idx.get(int(gl), bg_idx)
                cm[gt_idx, bg_idx] += 1

    if save_path:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names + ["background"],
                    yticklabels=class_names + ["background"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_name}")
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved to {save_path}")

    return cm


def draw_boxes(img, boxes, labels, scores, class_names, color, score_thresh=0.5):
    img = img.copy()
    for box, label, score in zip(boxes, labels, scores):
        if isinstance(score, torch.Tensor):
            score = score.item()
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box) if not isinstance(box, torch.Tensor) else map(int, box.cpu().numpy())
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        class_idx = int(label) - 1 if isinstance(label, (int, torch.Tensor)) else label
        if 0 <= class_idx < len(class_names):
            text = f"{class_names[class_idx]} {score:.2f}"
        else:
            text = f"cls{class_idx} {score:.2f}"
        cv2.putText(img, text, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return img


def plot_sample_detections(test_items, yolo_model, frcnn_model, class_names,
                           device, n=6, save_path=None):
    import random

    sample = random.sample(test_items, min(n, len(test_items)))
    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n))

    for i, item in enumerate(sample):
        img_path = item["image_path"]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        gt_img = draw_boxes(img, item["gt_boxes"], item["gt_labels"],
                           [1.0] * len(item["gt_boxes"]), class_names, (0, 200, 0), score_thresh=0.0)

        yolo_preds = predict_yolov8(yolo_model, img_path)
        y_boxes = [p["box"] for p in yolo_preds]
        y_labels = [p["label"] for p in yolo_preds]
        y_scores = [p["score"] for p in yolo_preds]
        y_img = draw_boxes(img, y_boxes, y_labels, y_scores, class_names, (0, 100, 255))

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        fr_preds = predict_fasterrcnn(frcnn_model, img_tensor, device)
        fr_boxes = [p["box"] for p in fr_preds]
        fr_labels = [p["label"] for p in fr_preds]
        fr_scores = [p["score"] for p in fr_preds]
        fr_img = draw_boxes(img, fr_boxes, fr_labels, fr_scores, class_names, (255, 50, 50))

        axes[i][0].imshow(gt_img)
        axes[i][0].set_title("Ground Truth")
        axes[i][0].axis("off")
        axes[i][1].imshow(y_img)
        axes[i][1].set_title("YOLOv8")
        axes[i][1].axis("off")
        axes[i][2].imshow(fr_img)
        axes[i][2].set_title("Faster R-CNN")
        axes[i][2].axis("off")

    plt.suptitle("Sample Detections: GT vs YOLOv8 vs Faster R-CNN", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Sample detections saved to {save_path}")


def plot_per_class_ap(yolo_aps, frcnn_aps, class_names, save_path):
    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, yolo_aps, width, label="YOLOv8", color="#1f77b4")
    ax.bar(x + width / 2, frcnn_aps, width, label="Faster R-CNN", color="#ff7f0e")

    ax.set_ylabel("AP @0.5")
    ax.set_title("Per-Class AP Comparison — YOLOv8 vs Faster R-CNN")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Per-class AP chart saved to {save_path}")
