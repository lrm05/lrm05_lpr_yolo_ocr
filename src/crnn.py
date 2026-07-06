"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务.

OCR 基线模型：
1. 轻量 CNN 抽取车牌图像特征。
2. 双向 LSTM 建模字符序列。
3. 使用 CTC Loss 进行无对齐训练。
"""

from __future__ import annotations

import torch
from torch import nn


class CrnnRecognizer(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.sequence_model = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        features = features.squeeze(2).permute(2, 0, 1)
        recurrent_features, _ = self.sequence_model(features)
        return self.classifier(recurrent_features)
