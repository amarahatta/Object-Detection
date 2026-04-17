import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm


def collate_fn(batch):
    return tuple(zip(*batch))


def build_fasterrcnn(num_classes):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    loss_components = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }
    loop = tqdm(data_loader, desc=f"Epoch {epoch}")

    for images, targets in loop:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        for key in loss_components:
            if key in loss_dict:
                loss_components[key] += loss_dict[key].item()

        loop.set_postfix(loss=losses.item())

    n = len(data_loader)
    avg_components = {k: v / n for k, v in loss_components.items()}
    return total_loss / n, avg_components


def val_one_epoch(model, data_loader, device, epoch):
    model.train()
    total_loss = 0

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc=f"Val Epoch {epoch}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    return total_loss / len(data_loader)
