"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务。
文件作用：这个文件用于对 OCR(CRNN + CTC) 与 LPRNet 在同一套车牌裁剪数据集上做统一评测，输出可直接写入分析文档的结果。
识别模型统一评测脚本：1. 加载同一测试集。2. 分别评测 CRNN 与 LPRNet。3. 输出 exact accuracy 和 char accuracy 对比结果。
运行方式：    python scripts/evaluate_recognizers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.crnn import CrnnRecognizer
from src.lprnet import LprNetRecognizer
from src.ocr_utils import Charset, PlateOcrDataset, ctc_collate_fn, greedy_decode


def load_charset(charset_path: Path) -> Charset:
    characters = [
        line.strip()
        for line in charset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return Charset(characters=characters)


def resolve_label_file(label_file: Path, split_name: str, image_root: Path, output_file: Path) -> Path:
    """把标签中的旧本地绝对路径改成当前环境真实可访问的图片路径。"""
    split_image_dir = image_root / split_name
    if not split_image_dir.exists():
        raise FileNotFoundError(f"未找到 OCR 图片目录: {split_image_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

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

    output_file.write_text("".join(resolved_lines), encoding="utf-8")
    return output_file


def build_model(model_name: str, charset: Charset, checkpoint: dict, device: str):
    normalized_name = model_name.lower()
    if normalized_name == "crnn":
        model = CrnnRecognizer(num_classes=charset.num_classes).to(device)
    elif normalized_name == "lprnet":
        model = LprNetRecognizer(num_classes=charset.num_classes).to(device)
    else:
        raise ValueError(f"Unsupported recognizer model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_model(model, loader: DataLoader, charset: Charset, device: str) -> dict[str, float]:
    exact_matches = 0
    char_matches = 0
    char_total = 0
    sample_total = 0

    with torch.no_grad():
        for images, texts, _, _ in loader:
            images = images.to(device)
            logits = model(images)
            predictions = greedy_decode(logits, charset)

            for predicted, target in zip(predictions, texts):
                sample_total += 1
                if predicted == target:
                    exact_matches += 1

                overlap = min(len(predicted), len(target))
                char_matches += sum(predicted[i] == target[i] for i in range(overlap))
                char_total += len(target)

    return {
        "sample_count": sample_total,
        "exact_accuracy": exact_matches / sample_total if sample_total else 0.0,
        "char_accuracy": char_matches / char_total if char_total else 0.0,
    }


def main() -> None:
    # ==================== 固定参数区 ====================
    dataset_dir = PROJECT_ROOT / "ocr"
    charset_path = dataset_dir / "charset.txt"
    test_label = dataset_dir / "test_label.txt"
    image_root = dataset_dir / "images"

    crnn_weights = PROJECT_ROOT / "runs_ocr" / "best.pt"
    lprnet_weights = PROJECT_ROOT / "runs_lprnet" / "best.pt"
    output_json = PROJECT_ROOT / "analysis" / "recognizer_compare_summary.json"
    resolved_test_label = PROJECT_ROOT / "analysis" / "test_label_resolved.txt"

    batch_size = 256
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ==================================================

    charset = load_charset(charset_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_test_label = resolve_label_file(
        label_file=test_label,
        split_name="test",
        image_root=image_root,
        output_file=resolved_test_label,
    )
    print(f"测试标签已修正为: {resolved_test_label}")

    test_dataset = PlateOcrDataset(resolved_test_label, charset=charset)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    crnn_checkpoint = torch.load(crnn_weights, map_location=device)
    lprnet_checkpoint = torch.load(lprnet_weights, map_location=device)

    crnn_model = build_model("crnn", charset, crnn_checkpoint, device=device)
    lprnet_model = build_model("lprnet", charset, lprnet_checkpoint, device=device)

    crnn_metrics = evaluate_model(crnn_model, test_loader, charset=charset, device=device)
    lprnet_metrics = evaluate_model(lprnet_model, test_loader, charset=charset, device=device)

    summary = {
        "dataset_dir": str(dataset_dir),
        "test_label": str(test_label),
        "ocr_crnn_ctc": crnn_metrics,
        "lprnet": lprnet_metrics,
    }

    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("统一识别评测完成")
    print(f"OCR(CRNN+CTC) exact_acc={crnn_metrics['exact_accuracy']:.4f} char_acc={crnn_metrics['char_accuracy']:.4f}")
    print(f"LPRNet exact_acc={lprnet_metrics['exact_accuracy']:.4f} char_acc={lprnet_metrics['char_accuracy']:.4f}")
    print(f"结果已保存到: {output_json}")


if __name__ == "__main__":
    main()
