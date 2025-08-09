import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from ultralytics import YOLO
import albumentations as A
import cv2
from tqdm import tqdm
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.tal import TaskAlignedAssigner
import random

import warnings
warnings.filterwarnings('ignore')


def apply_super_resolution_small_objects(image, **kwargs):
    scale_factor = random.choice([1.25, 1.5]) 
    lr = cv2.resize(image, (int(image.shape[1]/scale_factor), int(image.shape[0]/scale_factor)),
                   interpolation=cv2.INTER_AREA)
    lr = A.GaussNoise(var_limit=(0.5, 1.5), p=0.5)(image=lr)['image'] 
    return cv2.resize(lr, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

def safe_crop(image, target_size):  
    h, w = image.shape[:2]
    crop_h, crop_w = min(target_size[0], h), min(target_size[1], w)
    y = random.randint(0, h - crop_h)
    x = random.randint(0, w - crop_w)
    return image[y:y+crop_h, x:x+crop_w]

small_object_aug = A.Compose([
    A.RandomScale(scale_limit=(-0.3, 0.3)),  
    A.Lambda(name='safe_crop', image=lambda img, **kw: safe_crop(img, (700, 700)), p=0.7),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.15))

base_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(
        scale=(0.8, 1.2),  
        translate_percent=(-0.05, 0.05),  
        shear=(-3, 3),
        rotate=(-3, 3),
        keep_ratio=True,
        p=0.7
    ),
    A.OneOf([
        A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.2, val_shift_limit=0.1, p=0.5),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
    ], p=0.5),
    A.Lambda(name='super_res_small', image=apply_super_resolution_small_objects, p=0.2),
    small_object_aug
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.2))

final_transform = A.Compose([
    A.LongestMaxSize(max_size=1024, p=1.0), 
    base_transform
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.2))


def album_augmentations(batch):
    imgs = batch['img']
    orig_size = imgs.shape[2:]
    classes = batch['cls']
    labels_list = batch['bboxes']

    aug_imgs, aug_classes, aug_labels = [], [], []

    for i, (im, cls_ids, bboxes) in enumerate(zip(imgs, classes, labels_list)):
        im_np = im.permute(1, 2, 0).numpy()

        bboxes_np = bboxes.cpu().numpy().reshape(-1, 4)
        class_labels = cls_ids.cpu().numpy().astype(int)

        if len(bboxes_np) == 0:
            aug_imgs.append(im)
            aug_classes.append(cls_ids)
            aug_labels.append(bboxes)
            continue

        try:
            augmented = final_transform(
                image=im_np,
                bboxes=bboxes_np,
                class_labels=class_labels
            )

            if len(augmented['bboxes']) == 0:
                bboxes_out = torch.zeros((0, 4), dtype=torch.float32, device=im.device)
                classes_out = torch.zeros(0, dtype=torch.long, device=im.device)
            else:
                bboxes_out = torch.tensor(
                    augmented['bboxes'],
                    dtype=torch.float32,
                    device=im.device
                )
                classes_out = torch.tensor(
                    augmented['class_labels'],
                    dtype=torch.int32,
                    device=im.device
                )

            im_out = torch.from_numpy(augmented['image'].transpose(2, 0, 1)).to(im.device)

            im_out = F.interpolate(im_out.unsqueeze(0), size=orig_size, mode='bilinear').squeeze(0)

            h, w = augmented['image'].shape[:2]
            scale_x = orig_size[1] / w
            scale_y = orig_size[0] / h

            scaled_bboxes = []
            for bbox in augmented['bboxes']:
                x, y, bw, bh = bbox
                scaled_bboxes.append([
                    x * scale_x,
                    y * scale_y,
                    bw * scale_x,
                    bh * scale_y
                ])

            bboxes_out = torch.tensor(scaled_bboxes, dtype=torch.float32, device=im.device)
            classes_out = torch.tensor(augmented['class_labels'], dtype=torch.long, device=im.device)

            aug_imgs.append(im_out)
            aug_classes.append(classes_out)
            aug_labels.append(bboxes_out)

        except Exception as e:
            print(f"Augmentation error ({e}) occured on iter {i + 1}. Original image processed.")
            aug_imgs.append(im)
            aug_classes.append(cls_ids)
            aug_labels.append(bboxes)

    batch['img'] = torch.stack(aug_imgs)
    batch['cls'] = aug_classes
    batch['bboxes'] = aug_labels

    return batch


class CustomAugmentations:
    def __init__(self):
        pass

    def __call__(self, trainer):
        batch = next(iter(trainer.train_loader))
        aug_batch = album_augmentations(batch)

        return aug_batch


class LightAttention(nn.Module):
    def __init__(self, channels, reduction=10):
        super().__init__()
        self.channels = channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.LeakyReLU(),

            nn.Linear(channels // reduction, channels // reduction),
            nn.LeakyReLU(),

            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        self._init_weights()

    def _init_weights(self):
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='sigmoid')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(-1, self.channels))
        max_out = self.fc(self.max_pool(x).view(-1, self.channels))
        channel_att = (avg_out + max_out).view(-1, self.channels, 1, 1)

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.conv(torch.cat([avg_out, max_out], dim=1))

        return x * channel_att * spatial_att

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                 padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.activation = nn.LeakyReLU()

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.depthwise.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_uniform_(self.pointwise.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.activation(x)

class BiFPNAttentionBlock(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32))

        self.conv_p3 = nn.Sequential(
            SeparableConv2d(channels, channels),
        )
        self.conv_p4 = nn.Sequential(
            SeparableConv2d(channels, channels),
            LightAttention(channels)
        )
        self.conv_p5 = nn.Sequential(
            SeparableConv2d(channels, channels),
            LightAttention(channels)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.w1, 1.0)
        nn.init.constant_(self.w2, 1.0)

    def forward(self, p3, p4, p5):
        w1 = F.softmax(self.w1, dim=0)
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        p4_td = w1[0] * p4 + w1[1] * p5_up

        p4_up = F.interpolate(p4_td, size=p3.shape[2:], mode='nearest')
        p3_td = w1[0] * p3 + w1[1] * p4_up

        w2 = F.softmax(self.w2, dim=0)
        p3_down = F.max_pool2d(p3_td, kernel_size=2)
        p4_out = w2[0] * p4 + w2[1] * p4_td + w2[2] * p3_down

        p4_down = F.max_pool2d(p4_out, kernel_size=2)
        p5_out = w2[0] * p5 + w2[1] * F.max_pool2d(p5_up, kernel_size=2) + w2[2] * p4_down

        return self.conv_p3(p3_td), self.conv_p4(p4_out), self.conv_p5(p5_out)

class CustomBiFPN(nn.Module):
    def __init__(self, in_channels=[256, 512, 2048], out_channels=256, num_blocks=2):
        super().__init__()

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            ) for ch in in_channels
        ])

        self.bifpn = nn.Sequential(*[
            BiFPNAttentionBlock(out_channels) for _ in range(num_blocks)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.proj:
            nn.init.xavier_uniform_(m[0].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m[1].weight, 1.0)
            nn.init.constant_(m[1].bias, 0.0)

    def forward(self, features):
        p3, p4, p5 = [proj(x) for proj, x in zip(self.proj, features)]
        for bifpn_block in self.bifpn:
            p3, p4, p5 = bifpn_block(p3, p4, p5)
        return [p3, p4, p5]


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=1.5, reduction='mean'):
        super().__init__()
        self.register_buffer('alpha', torch.as_tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
        self.reduction = reduction
        self.epsilon = 1e-6  

    def forward(self, inputs, targets):
        if inputs.numel() == 0: 
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        log_softmax = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(log_softmax, targets, reduction='none')

        pt = torch.exp(-ce_loss) + self.epsilon

        focal_term = ((1 - pt) ** self.gamma)
        alpha_t = self.alpha.gather(0, targets)
        weighted_loss = alpha_t * focal_term * ce_loss

        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        return weighted_loss


class CustomLoss:
    def __init__(self, model, weights):
        self.model = model
        num_classes = len(model.model.names)  
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=num_classes, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(num_classes)
        self.cls_loss = FocalLoss(alpha=weights, gamma=1.5)

    def __call__(self, preds, batch):
        pred_distri, pred_scores = preds[1][:2]
        targets = torch.cat([batch['batch_idx'].view(-1, 1),
                           batch['cls'].view(-1, 1),
                           batch['bboxes']], 1)

        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_distri.detach(),
            batch['anchors'],
            targets
        )

        loss_bbox = self.bbox_loss(pred_distri, target_bboxes, batch['anchors'], fg_mask)

        loss_cls = self.cls_loss(
            pred_scores[fg_mask],
            target_scores[fg_mask].argmax(1) 
        )

        loss = loss_bbox[0] + loss_bbox[1] + loss_cls 
        return loss, torch.cat((loss_bbox, loss_cls.unsqueeze(0)))


class PresenceHead(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        hidden = max(8, in_channels // reduction)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False), 
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.block:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.)

    def forward(self, x):
        x = self.block(x)              
        x = self.pool(x).view(x.size(0), -1)  
        x = self.fc(x)                  
        return self.sigmoid(x).squeeze(1)  


class CustomYOLO(YOLO):
    def __init__(self, *args, presence_in_ch=64, **kwargs):
        super().__init__(*args, **kwargs)

        self.presence_head = PresenceHead(presence_in_ch).to(self.device)
        self.presence_loss_weight = 1.0
        self.presence_criterion = nn.BCELoss()

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


model = CustomYOLO('yolov8m.pt')

alpha = torch.tensor([0.1, 0.8, 0.7, 0.25, 0.5]).to(model.device)
model.loss = CustomLoss(model, alpha)
model.loss.cls_loss.alpha = alpha.to(model.device)

model.model.neck = CustomBiFPN([192, 384, 576], 192, num_blocks=3).to(model.device)

train_args = {
    'data': "/home/user/dataset/data.yaml",
    'epochs': 100,
    'batch': 16,
    'imgsz': 640,
    'optimizer': 'AdamW',
    'lr0': 1e-3,
    'momentum': 0.95,
    'weight_decay': 0.05,
    'warmup_epochs': 5,
    'warmup_momentum': 0.8,
    'augment': True,
    'device': '0,1',
    'amp': True,
    'workers': 4
}

model.train(
    seed=42,
    **train_args
)