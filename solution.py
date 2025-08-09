import numpy as np
import os
import hashlib
from typing import List, Union
import torch
from ultralytics import YOLO
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class CustomYOLO(YOLO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_optimizers(self):
        optimizer_config = {
            'type': 'AdamW',
            'base_lr': 0.001,
            'momentum': 0.95,
            'backbone_lr_ratio': 0.2,
            'bifpn_lr_ratio': 0.7,
            'head_lr_ratio': 1.0,
            'bias_lr_ratio': 2.5,
            'weight_decay': 0.01,
            'weight_decay_bias': 0.0,
            'clip_grad_norm': 4.0
        }

        scheduler_config = {
            'scheduler': 'CosineAnnealingLR',
            'T_max': 100,
            'eta_min': 0.00001,
            'warmup_epochs': 10,
            'warmup_lr_init': 1e-6,
            'warmup_momentum': 0.82
        }

        param_groups = {
            'backbone': {'params': [], 'lr_ratio': optimizer_config['backbone_lr_ratio']},
            'bifpn': {'params': [], 'lr_ratio': optimizer_config['bifpn_lr_ratio']},
            'head': {'params': [], 'lr_ratio': optimizer_config['head_lr_ratio']},
            'biases': {'params': [], 'lr_ratio': optimizer_config['bias_lr_ratio'], 'weight_decay': 0.0}
        }

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name:
                param_groups['biases']['params'].append(param)
            elif 'backbone' in name:
                param_groups['backbone']['params'].append(param)
            elif 'bifpn' in name.lower():
                param_groups['bifpn']['params'].append(param)
            else:
                param_groups['head']['params'].append(param)

        optimizer_groups = []
        for group in param_groups.values():
            lr = optimizer_config['base_lr'] * group.get('lr_ratio', 1.0)
            wd = group.get('weight_decay', optimizer_config['weight_decay'])
            optimizer_groups.append({
                'params': group['params'],
                'lr': lr,
                'weight_decay': wd
            })

        optimizer = AdamW(
            optimizer_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            foreach=True
        )

        lr_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max'],
            eta_min=scheduler_config['eta_min']
        )

        if scheduler_config['warmup_epochs'] > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=scheduler_config['warmup_lr_init'] / optimizer_config['base_lr'],
                end_factor=1.0,
                total_iters=scheduler_config['warmup_epochs']
            )
            lr_scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, lr_scheduler],
                milestones=[scheduler_config['warmup_epochs']]
            )

        return optimizer, lr_scheduler


model = CustomYOLO("best.pt")


def non_maximum_suppression_area(boxes, scores, iou_threshold=0.15):
    if len(boxes) == 0:
        return []

    boxes_xyxy = []
    areas = []
    for box in boxes:
        xc, yc, w, h = box
        xc = xc.cpu().item() if isinstance(xc, torch.Tensor) else xc
        yc = yc.cpu().item() if isinstance(yc, torch.Tensor) else yc
        w = w.cpu().item() if isinstance(w, torch.Tensor) else w
        h = h.cpu().item() if isinstance(h, torch.Tensor) else h

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        boxes_xyxy.append([x1, y1, x2, y2])

        areas.append(w * h)

    boxes_xyxy = np.array(boxes_xyxy)
    areas = np.array(areas)
    scores = np.array([s.cpu().item() if isinstance(s, torch.Tensor) else s for s in scores])

    indices = np.argsort(areas)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        current_box = boxes_xyxy[current]
        other_boxes = boxes_xyxy[indices[1:]]

        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union = current_area + other_areas - intersection

        iou = intersection / (union + 1e-6)

        indices = indices[np.where(iou <= iou_threshold)[0] + 1]

    return keep


def sliding_window_detection(image: np.ndarray, window_size=640, overlap=0.5, save_dir="results_window"):
    h, w = image.shape[:2]
    stride = int(window_size * (1 - overlap))

    all_boxes = []
    all_scores = []

    os.makedirs(save_dir, exist_ok=True)

    window_count = 0
    image_hash = hashlib.md5(image.tobytes()).hexdigest()[:8]

    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            window_count += 1
            window = image[y:y + window_size, x:x + window_size]
            window_hash = hashlib.md5(window.tobytes()).hexdigest()[:6]

            result = model.predict(source=window, imgsz=640, device=0, verbose=False, conf=0.4)

            result[0].save(filename=os.path.join(save_dir, f"{image_hash}_{window_count}_{window_hash}.jpg"))

            if len(result) > 0:
                res = result[0]
                if res.boxes is not None and len(res.boxes) > 0:
                    for box in res.boxes:
                        xywhn = box.xywhn[0]
                        xc_norm = xywhn[0].cpu().item()
                        yc_norm = xywhn[1].cpu().item()
                        w_norm = xywhn[2].cpu().item()
                        h_norm = xywhn[3].cpu().item()
                        conf = box.conf[0].cpu().item()

                        xc_orig = (xc_norm * window_size + x) / w
                        yc_orig = (yc_norm * window_size + y) / h
                        w_orig = w_norm * window_size / w
                        h_orig = h_norm * window_size / h

                        xc_orig = np.clip(xc_orig, 0.0, 1.0)
                        yc_orig = np.clip(yc_orig, 0.0, 1.0)
                        w_orig = np.clip(w_orig, 0.0, 1.0)
                        h_orig = np.clip(h_orig, 0.0, 1.0)

                        if xc_orig + w_orig / 2 > 1.0:
                            xc_orig = 1.0 - w_orig / 2
                        if yc_orig + h_orig / 2 > 1.0:
                            yc_orig = 1.0 - h_orig / 2
                        if xc_orig - w_orig / 2 < 0.0:
                            xc_orig = w_orig / 2
                        if yc_orig - h_orig / 2 < 0.0:
                            yc_orig = h_orig / 2

                        all_boxes.append([xc_orig, yc_orig, w_orig, h_orig])
                        all_scores.append(conf)

    return all_boxes, all_scores


def infer_image_bbox(image: np.ndarray) -> List[dict]:
    all_boxes = []
    all_scores = []

    sw_boxes, sw_scores = sliding_window_detection(image, window_size=640, overlap=0.1)
    all_boxes.extend(sw_boxes)
    all_scores.extend(sw_scores)

    if len(all_boxes) > 0:
        keep_indices = non_maximum_suppression_area(all_boxes, all_scores, iou_threshold=0.15)
        res_list = []

        for idx in keep_indices:
            xc, yc, w_box, h_box = all_boxes[idx]
            conf = all_scores[idx]

            if conf >= 0.3:
                xc = np.clip(xc, 0.0, 1.0)
                yc = np.clip(yc, 0.0, 1.0)
                w_box = np.clip(w_box, 0.0, 1.0)
                h_box = np.clip(h_box, 0.0, 1.0)

                if xc + w_box / 2 > 1.0:
                    xc = 1.0 - w_box / 2
                if yc + h_box / 2 > 1.0:
                    yc = 1.0 - h_box / 2
                if xc - w_box / 2 < 0.0:
                    xc = w_box / 2
                if yc - h_box / 2 < 0.0:
                    yc = h_box / 2

                res_list.append({
                    'xc': xc,
                    'yc': yc,
                    'w': w_box,
                    'h': h_box,
                    'label': 0,
                    'score': conf
                })

        return res_list

    return []


def predict(images: Union[List[np.ndarray], np.ndarray]) -> List[List[dict]]:
    results = []
    if isinstance(images, np.ndarray):
        images = [images]

    for image in images:
        results.append(infer_image_bbox(image))

    return results
