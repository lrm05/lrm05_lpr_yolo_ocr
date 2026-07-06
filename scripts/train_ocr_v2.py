"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务。
OCR v2 training entry using merged OCR dataset.

Run:
    python scripts/train_ocr_v2.py
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

from src.crnn import CrnnRecognizer
from src.ocr_utils import Charset, PlateOcrDataset, ctc_collate_fn, greedy_decode


def load_charset(charset_path: Path) -> Charset:
    characters = [
        line.strip()
        for line in charset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return Charset(characters=characters)


def resolve_label_file(label_file: Path, split_name: str, image_root: Path, output_file: Path) -> Path:
    split_image_dir = image_root / split_name
    if not split_image_dir.exists():
        raise FileNotFoundError(f"OCR image directory not found: {split_image_dir}")

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
        raise FileNotFoundError(f"Missing images in label file {label_file}:\n{preview}")

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
    # Fixed parameters
    dataset_dir = PROJECT_ROOT / "ocr_v2"
    charset_path = dataset_dir / "charset.txt"
    train_label = dataset_dir / "train_label.txt"
    val_label = dataset_dir / "val_label.txt"
    image_root = dataset_dir / "images"
    output_dir = PROJECT_ROOT / "runs_ocr_v2"

    epochs = 100
    batch_size = 128
    num_workers = 4
    learning_rate = 5e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"OCR v2 dataset directory not found: {dataset_dir}\n"
            "Run python scripts/prepare_ocr_v2_dataset.py first"
        )

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

    print(f"Resolved train label: {resolved_train_label}")
    print(f"Resolved val label: {resolved_val_label}")

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

    model = CrnnRecognizer(num_classes=charset.num_classes).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

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
                },
                output_dir / "best.pt",
            )

    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_exact_accuracy": best_exact_accuracy,
                "best_char_accuracy": best_char_accuracy,
                "dataset_dir": str(dataset_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Training finished, best weights saved to: {output_dir / 'best.pt'}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best exact accuracy: {best_exact_accuracy:.4f}")
    print(f"Best char accuracy: {best_char_accuracy:.4f}")


if __name__ == "__main__":
    main()
