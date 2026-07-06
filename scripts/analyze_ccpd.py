"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务.

数据集分析脚本：
1. 统计 CCPD 样本总数、有效样本数和车牌长度分布。
2. 统计亮度、模糊度、字符分布和省份分布。
3. 输出 JSON 和 Markdown 摘要，作为算法设计文档输入。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ccpd_utils import iter_image_paths, parse_ccpd_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze CCPD-style dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to CCPD image directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save analysis outputs.")
    return parser


def brightness_bucket(brightness: int) -> str:
    if brightness <= 80:
        return "night_like"
    if brightness <= 130:
        return "twilight_like"
    return "day_like"


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    valid_samples = 0
    invalid_samples = 0
    plate_length_counter: Counter[int] = Counter()
    province_counter: Counter[str] = Counter()
    char_counter: Counter[str] = Counter()
    brightness_values: list[int] = []
    blur_values: list[int] = []
    brightness_bucket_counter: Counter[str] = Counter()

    for image_path in iter_image_paths(args.dataset_dir):
        total_images += 1
        sample = parse_ccpd_path(image_path)
        if sample is None:
            invalid_samples += 1
            continue

        valid_samples += 1
        plate_length_counter[sample.plate_length] += 1
        province_counter[sample.plate_text[0]] += 1
        char_counter.update(sample.plate_text[1:])
        brightness_values.append(sample.brightness)
        blur_values.append(sample.blur)
        brightness_bucket_counter[brightness_bucket(sample.brightness)] += 1

    summary = {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "total_images": total_images,
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "valid_ratio": round(valid_samples / total_images, 4) if total_images else 0.0,
        "plate_length_distribution": dict(sorted(plate_length_counter.items())),
        "top_10_provinces": province_counter.most_common(10),
        "top_10_characters": char_counter.most_common(10),
        "brightness": {
            "min": min(brightness_values) if brightness_values else None,
            "max": max(brightness_values) if brightness_values else None,
            "mean": round(sum(brightness_values) / len(brightness_values), 2) if brightness_values else None,
            "bucket_distribution": dict(brightness_bucket_counter),
        },
        "blur": {
            "min": min(blur_values) if blur_values else None,
            "max": max(blur_values) if blur_values else None,
            "mean": round(sum(blur_values) / len(blur_values), 2) if blur_values else None,
        },
    }

    summary_json = args.output_dir / "ccpd_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_md = args.output_dir / "ccpd_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# CCPD 数据集分析摘要",
                "",
                f"- 数据目录: `{summary['dataset_dir']}`",
                f"- 图片总数: `{summary['total_images']}`",
                f"- 有效样本数: `{summary['valid_samples']}`",
                f"- 无效样本数: `{summary['invalid_samples']}`",
                f"- 有效率: `{summary['valid_ratio']}`",
                f"- 车牌长度分布: `{summary['plate_length_distribution']}`",
                f"- 亮度分桶: `{summary['brightness']['bucket_distribution']}`",
                f"- 亮度均值: `{summary['brightness']['mean']}`",
                f"- 模糊度均值: `{summary['blur']['mean']}`",
                f"- 省份 Top10: `{summary['top_10_provinces']}`",
                f"- 字符 Top10: `{summary['top_10_characters']}`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"分析完成，JSON 已保存到: {summary_json}")
    print(f"分析完成，Markdown 已保存到: {summary_md}")


if __name__ == "__main__":
    main()
