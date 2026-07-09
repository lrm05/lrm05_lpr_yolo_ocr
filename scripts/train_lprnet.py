"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务。
文件作用：这个文件用于训练 LPRNet 车牌识别基线模型，和 OCR(CRNN + CTC) 使用同一套 ocr 数据集与同口径指标。
LPRNet 识别训练入口：1. 读取 ocr/train_label.txt 与 ocr/val_label.txt。2. 训练 LPRNet + CTC 基线模型。3. 输出 best.pt 和 metrics.json 供对比分析文档使用。
运行方式：    python scripts/train_lprnet.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    """把标签中的旧本地绝对路径改成云端当前可访问的真实路径。"""
    split_image_dir = image_root / split_name
    if not split_image_dir.exists():
        raise FileNotFoundError(f"未找到 OCR 图片目录: {split_image_dir}")

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


def evaluate(model: nn.Module, loader: DataLoader, charset: Charset, device: str) -> dict[str, float]:
    model.eval()
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
        "exact_accuracy": exact_matches / sample_total if sample_total else 0.0,
        "char_accuracy": char_matches / char_total if char_total else 0.0,
    }


def main() -> None:
    # ==================== 固定参数区 ====================
    # LPRNet 基线与 OCR v1 使用完全相同的数据切分，确保对比口径一致。
    dataset_dir = PROJECT_ROOT / "ocr"
    charset_path = dataset_dir / "charset.txt"
    train_label = dataset_dir / "train_label.txt"
    val_label = dataset_dir / "val_label.txt"
    image_root = dataset_dir / "images"
    output_dir = PROJECT_ROOT / "runs_lprnet"

    epochs = 80
    batch_size = 128
    num_workers = 4
    learning_rate = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ==================================================

    output_dir.mkdir(parents=True, exist_ok=True)
    charset = load_charset(charset_path)

    resolved_train_label = resolve_label_file(
        label_file=train_label,
        split_name="train",
        image_root=image_root,
        output_file=output_dir / "train_label_resolved.txt",
    )
    resolved_val_label = resolve_label_file(
        label_file=val_label,
        split_name="val",
        image_root=image_root,
        output_file=output_dir / "val_label_resolved.txt",
    )

    print(f"训练标签已修正为: {resolved_train_label}")
    print(f"验证标签已修正为: {resolved_val_label}")

    train_dataset = PlateOcrDataset(resolved_train_label, charset=charset)
    val_dataset = PlateOcrDataset(resolved_val_label, charset=charset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = LprNetRecognizer(num_classes=charset.num_classes).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_exact_accuracy = 0.0
    best_char_accuracy = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, _, targets, target_lengths in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(images)
            log_probs = logits.log_softmax(dim=2)
            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=log_probs.size(0),
                dtype=torch.long,
                device=device,
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        metrics = evaluate(model, val_loader, charset=charset, device=device)
        average_loss = running_loss / max(1, len(train_loader))

        print(
            f"Epoch {epoch:03d} | "
            f"loss={average_loss:.4f} | "
            f"exact_acc={metrics['exact_accuracy']:.4f} | "
            f"char_acc={metrics['char_accuracy']:.4f}"
        )

        if metrics["exact_accuracy"] >= best_exact_accuracy:
            best_exact_accuracy = metrics["exact_accuracy"]
            best_char_accuracy = metrics["char_accuracy"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "charset": charset.characters,
                    "metrics": metrics,
                    "epoch": epoch,
                    "model_name": "LPRNet",
                },
                output_dir / "best.pt",
            )

    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_name": "LPRNet",
                "dataset_dir": str(dataset_dir),
                "best_epoch": best_epoch,
                "best_exact_accuracy": best_exact_accuracy,
                "best_char_accuracy": best_char_accuracy,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"训练完成，最佳权重保存到: {output_dir / 'best.pt'}")
    print(f"最佳轮次: {best_epoch}")
    print(f"最佳整牌准确率: {best_exact_accuracy:.4f}")
    print(f"最佳字符准确率: {best_char_accuracy:.4f}")


if __name__ == "__main__":
    main()
