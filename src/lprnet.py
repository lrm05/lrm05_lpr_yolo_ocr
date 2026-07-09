"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务。
文件作用：这个文件用于定义 LPRNet 车牌识别模型结构，作为与 OCR(CRNN + CTC) 同口径对比的基线识别模型。
LPRNet 基线模型：1. 使用轻量卷积网络提取车牌图像特征。2. 在宽度方向上聚合时序特征。3. 输出字符分类序列并配合 CTC Loss 训练。
"""

from __future__ import annotations

import torch
from torch import nn


class SmallBasicBlock(nn.Module):
    """LPRNet 常见的小型卷积块，用来在较低计算量下增强局部特征表达。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        mid_channels = out_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LprNetRecognizer(nn.Module):
    """轻量 LPRNet 基线实现，输出形状为 [time_steps, batch_size, num_classes]。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            SmallBasicBlock(64, 128),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1),
            SmallBasicBlock(128, 256),
            SmallBasicBlock(256, 256),
            nn.MaxPool2d(kernel_size=3, stride=(2, 1), padding=1),
            nn.Dropout(0.3),
            nn.Conv2d(256, 256, kernel_size=(4, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(256, num_classes, kernel_size=(1, 13), stride=1, padding=(0, 6)),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        # 将高度方向做平均，保留宽度方向作为时序维度，输出给 CTC。
        logits = features.mean(dim=2)
        return logits.permute(2, 0, 1).contiguous()
