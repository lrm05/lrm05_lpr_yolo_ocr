"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务.

CCPD 数据集解析工具：
1. 从文件名解析车牌框、四点坐标、亮度、模糊度和字符标签。
2. 为 YOLO 检测训练和 OCR 识别训练提供统一样本对象。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


PROVINCES = [
    "皖",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "京",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
]

ALPHANUMERIC = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
DEFAULT_CHARSET = PROVINCES + ALPHANUMERIC


@dataclass(frozen=True)
class CcpdSample:
    image_path: Path
    filename: str
    area: int
    hor_angle: int
    ver_angle: int
    bbox: tuple[int, int, int, int]
    polygon: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]
    plate_text: str
    brightness: int
    blur: int

    @property
    def plate_length(self) -> int:
        return len(self.plate_text)


def iter_image_paths(dataset_dir: Path) -> Iterable[Path]:
    for path in sorted(dataset_dir.glob("*.jpg")):
        yield path


def decode_plate(label_codes: list[int]) -> str | None:
    if len(label_codes) < 7:
        return None

    province_index = label_codes[0]
    if province_index < 0 or province_index >= len(PROVINCES):
        return None

    text = [PROVINCES[province_index]]
    for code in label_codes[1:]:
        if code < 0 or code >= len(ALPHANUMERIC):
            return None
        text.append(ALPHANUMERIC[code])
    return "".join(text)


def parse_ccpd_path(image_path: Path) -> CcpdSample | None:
    if image_path.suffix.lower() != ".jpg":
        return None

    parts = image_path.stem.split("-")
    if len(parts) != 7:
        return None

    try:
        area = int(parts[0])
        hor_angle, ver_angle = [int(item) for item in parts[1].split("_")]

        top_left, bottom_right = parts[2].split("_")
        x1, y1 = [int(item) for item in top_left.split("&")]
        x2, y2 = [int(item) for item in bottom_right.split("&")]
        bbox = (x1, y1, x2, y2)

        polygon = []
        for point in parts[3].split("_"):
            point_x, point_y = [int(item) for item in point.split("&")]
            polygon.append((point_x, point_y))
        if len(polygon) != 4:
            return None

        label_codes = [int(item) for item in parts[4].split("_")]
        plate_text = decode_plate(label_codes)
        if plate_text is None:
            return None

        brightness = int(parts[5])
        blur = int(parts[6])
    except ValueError:
        return None

    return CcpdSample(
        image_path=image_path,
        filename=image_path.name,
        area=area,
        hor_angle=hor_angle,
        ver_angle=ver_angle,
        bbox=bbox,
        polygon=tuple(polygon),
        plate_text=plate_text,
        brightness=brightness,
        blur=blur,
    )


def clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(1, min(width, x2))
    y2 = max(1, min(height, y2))
    return x1, y1, x2, y2


def expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    margin_ratio: float = 0.05,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    margin_x = int(box_width * margin_ratio)
    margin_y = int(box_height * margin_ratio)
    return clamp_bbox((x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y), width, height)


def xyxy_to_yolo(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    center_x = ((x1 + x2) / 2.0) / width
    center_y = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return center_x, center_y, box_width, box_height
