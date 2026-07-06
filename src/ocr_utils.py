"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务.

OCR 数据集与解码工具：
1. 读取裁剪后的车牌图片和标签文件。
2. 负责字符编码、CTC 训练标签整理和推理解码。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Charset:
    characters: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.characters) + 1

    def encode(self, text: str) -> list[int]:
        mapping = {char: index + 1 for index, char in enumerate(self.characters)}
        return [mapping[char] for char in text]

    def decode(self, indices: list[int]) -> str:
        return "".join(self.characters[index - 1] for index in indices if index > 0)


class PlateOcrDataset(Dataset):
    def __init__(self, label_file: Path, charset: Charset, image_height: int = 48, image_width: int = 168):
        self.label_file = label_file
        self.charset = charset
        self.image_height = image_height
        self.image_width = image_width
        self.samples = self._load_samples(label_file)

    @staticmethod
    def _load_samples(label_file: Path) -> list[tuple[Path, str]]:
        samples: list[tuple[Path, str]] = []
        for raw_line in label_file.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            image_path, text = raw_line.split("\t", maxsplit=1)
            samples.append((Path(image_path), text.strip()))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, torch.Tensor]:
        image_path, text = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"无法读取 OCR 图像: {image_path}")

        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        image_tensor = (image_tensor - 0.5) / 0.5
        encoded = torch.tensor(self.charset.encode(text), dtype=torch.long)
        return image_tensor, text, encoded


def ctc_collate_fn(batch: list[tuple[torch.Tensor, str, torch.Tensor]]):
    images, texts, encoded_sequences = zip(*batch)
    image_tensor = torch.stack(images, dim=0)
    target_lengths = torch.tensor([seq.numel() for seq in encoded_sequences], dtype=torch.long)
    targets = torch.cat(encoded_sequences, dim=0)
    return image_tensor, list(texts), targets, target_lengths


def greedy_decode(logits: torch.Tensor, charset: Charset) -> list[str]:
    predictions = logits.argmax(dim=2).transpose(0, 1)
    decoded_texts: list[str] = []

    for sequence in predictions:
        deduplicated: list[int] = []
        previous = -1
        for value in sequence.tolist():
            if value != previous and value != 0:
                deduplicated.append(value)
            previous = value
        decoded_texts.append(charset.decode(deduplicated))

    return decoded_texts
