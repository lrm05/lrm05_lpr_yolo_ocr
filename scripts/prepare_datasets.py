"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务.

训练数据准备脚本：
1. 从 CCPD 文件名解析车牌框和车牌文本。
2. 生成 YOLO 检测数据集。
3. 生成 OCR 识别裁剪图与标签文件。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ccpd_utils import DEFAULT_CHARSET, clamp_bbox, expand_bbox, iter_image_paths, parse_ccpd_path, xyxy_to_yolo


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare YOLO and OCR datasets from CCPD images.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to CCPD image directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("prepared_data"), help="Directory to save processed datasets.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--crop-margin-ratio", type=float, default=0.05, help="Extra OCR crop margin.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for quick smoke tests.")
    return parser


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def split_samples(samples: list, train_ratio: float, val_ratio: float):
    train_end = int(len(samples) * train_ratio)
    val_end = train_end + int(len(samples) * val_ratio)
    return {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }


def save_yolo_label(label_path: Path, normalized_bbox: tuple[float, float, float, float]) -> None:
    x_center, y_center, box_width, box_height = normalized_bbox
    label_path.write_text(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n", encoding="utf-8")


def main() -> None:
    args = build_argparser().parse_args()
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio 必须小于 1.0")

    ensure_dir(args.output_dir)
    parsed_samples = []
    for image_path in iter_image_paths(args.dataset_dir):
        sample = parse_ccpd_path(image_path)
        if sample is not None:
            parsed_samples.append(sample)
        if args.max_samples and len(parsed_samples) >= args.max_samples:
            break

    random.seed(args.seed)
    random.shuffle(parsed_samples)
    split_mapping = split_samples(parsed_samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    yolo_root = args.output_dir / "yolo"
    ocr_root = args.output_dir / "ocr"

    for split_name in ("train", "val", "test"):
        ensure_dir(yolo_root / "images" / split_name)
        ensure_dir(yolo_root / "labels" / split_name)
        ensure_dir(ocr_root / "images" / split_name)

    ocr_label_files = {"train": [], "val": [], "test": []}
    split_manifest = {}

    for split_name, samples in split_mapping.items():
        split_manifest[split_name] = []
        for sample in samples:
            image = cv2.imread(str(sample.image_path))
            if image is None:
                continue

            height, width = image.shape[:2]
            bbox = clamp_bbox(sample.bbox, width=width, height=height)
            normalized_bbox = xyxy_to_yolo(bbox, width=width, height=height)

            dst_image_path = yolo_root / "images" / split_name / sample.filename
            label_path = yolo_root / "labels" / split_name / f"{sample.image_path.stem}.txt"
            link_or_copy(sample.image_path, dst_image_path)
            save_yolo_label(label_path, normalized_bbox)

            crop_bbox = expand_bbox(bbox, width=width, height=height, margin_ratio=args.crop_margin_ratio)
            x1, y1, x2, y2 = crop_bbox
            crop = image[y1:y2, x1:x2]
            crop_path = ocr_root / "images" / split_name / sample.filename
            cv2.imwrite(str(crop_path), crop)
            ocr_label_files[split_name].append(f"{crop_path.resolve()}\t{sample.plate_text}\n")

            split_manifest[split_name].append(
                {
                    "filename": sample.filename,
                    "plate_text": sample.plate_text,
                    "brightness": sample.brightness,
                    "blur": sample.blur,
                    "bbox": list(bbox),
                }
            )

    for split_name, lines in ocr_label_files.items():
        (ocr_root / f"{split_name}_label.txt").write_text("".join(lines), encoding="utf-8")

    (ocr_root / "charset.txt").write_text("\n".join(DEFAULT_CHARSET) + "\n", encoding="utf-8")
    (args.output_dir / "yolo_ccpd.yaml").write_text(
        "\n".join(
            [
                f"path: {yolo_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                "  0: plate",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"数据准备完成，输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
