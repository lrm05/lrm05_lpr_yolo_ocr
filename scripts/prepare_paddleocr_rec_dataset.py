"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务。
文件作用：这个文件用于把当前 ocr 识别数据集转换成 PaddleOCR 文本识别训练所需的标签格式，并保持与 OCR、LPRNet 使用同一套数据集。
PaddleOCR 识别数据准备入口：1. 读取 ocr/train_label.txt、val_label.txt、test_label.txt。2. 自动修正旧路径。3. 生成 PaddleOCR 可直接使用的标签文件和字符字典。
运行方式：    python scripts/prepare_paddleocr_rec_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_charset(charset_path: Path) -> list[str]:
    return [
        line.strip()
        for line in charset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_label_file(label_file: Path, split_name: str, image_root: Path, output_file: Path) -> tuple[Path, int]:
    """把旧本地路径修正成当前环境可访问的图片路径，输出为 PaddleOCR 识别标签格式。"""
    split_image_dir = image_root / split_name
    if not split_image_dir.exists():
        raise FileNotFoundError(f"未找到识别图片目录: {split_image_dir}")

    resolved_lines: list[str] = []
    missing_images: list[str] = []

    for raw_line in label_file.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue

        image_path_str, text = raw_line.split("\t", maxsplit=1)
        image_path_str = image_path_str.strip()

        original_path = Path(image_path_str)
        if original_path.exists():
            candidate_path = original_path
        else:
            filename = image_path_str.replace("\\", "/").split("/")[-1]
            candidate_path = split_image_dir / filename

        if not candidate_path.exists():
            missing_images.append(str(candidate_path))
            continue

        resolved_lines.append(f"{candidate_path}\t{text.strip()}\n")

    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(
            f"{label_file} 中有图片在当前环境找不到，前 10 个示例：\n{preview}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("".join(resolved_lines), encoding="utf-8")
    return output_file, len(resolved_lines)


def main() -> None:
    # ==================== 固定参数区 ====================
    dataset_dir = PROJECT_ROOT / "ocr"
    image_root = dataset_dir / "images"
    charset_path = dataset_dir / "charset.txt"

    train_label = dataset_dir / "train_label.txt"
    val_label = dataset_dir / "val_label.txt"
    test_label = dataset_dir / "test_label.txt"

    output_dir = PROJECT_ROOT / "paddleocr_rec"
    train_out = output_dir / "train_label.txt"
    val_out = output_dir / "val_label.txt"
    test_out = output_dir / "test_label.txt"
    dict_out = output_dir / "ppocr_keys_v1.txt"
    summary_out = output_dir / "prepare_summary.json"
    # ==================================================

    output_dir.mkdir(parents=True, exist_ok=True)

    charset = load_charset(charset_path)
    dict_out.write_text("\n".join(charset) + "\n", encoding="utf-8")

    resolved_train, train_count = resolve_label_file(train_label, "train", image_root, train_out)
    resolved_val, val_count = resolve_label_file(val_label, "val", image_root, val_out)
    resolved_test, test_count = resolve_label_file(test_label, "test", image_root, test_out)

    summary = {
        "source_dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "charset_file": str(dict_out),
        "train_label": str(resolved_train),
        "val_label": str(resolved_val),
        "test_label": str(resolved_test),
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("PaddleOCR 识别数据准备完成")
    print(f"训练标签: {resolved_train}")
    print(f"验证标签: {resolved_val}")
    print(f"测试标签: {resolved_test}")
    print(f"字符字典: {dict_out}")
    print(f"训练样本数: {train_count}")
    print(f"验证样本数: {val_count}")
    print(f"测试样本数: {test_count}")


if __name__ == "__main__":
    main()
