"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务。
Prepare OCR v2 dataset by merging current OCR data with sampled CBLPRD data.

Run:
    python scripts/prepare_ocr_v2_dataset.py
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_charset(charset_path: Path) -> list[str]:
    return [
        line.strip()
        for line in charset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_label_file(label_file: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for raw_line in label_file.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        image_path_str, text = raw_line.split("\t", maxsplit=1)
        samples.append((Path(image_path_str.strip()), text.strip()))
    return samples


def parse_cblprd_label_file(label_file: Path, image_root: Path) -> list[tuple[Path, str, str]]:
    samples: list[tuple[Path, str, str]] = []
    for raw_line in label_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split(maxsplit=2)
        if len(parts) < 2:
            continue

        relative_image_path = parts[0].strip()
        plate_text = parts[1].strip()
        plate_type = parts[2].strip() if len(parts) >= 3 else ""
        image_path = image_root / Path(relative_image_path).name
        samples.append((image_path, plate_text, plate_type))
    return samples


def balanced_sample_by_prefix(
    samples: list[tuple[Path, str, str]],
    target_count: int,
    seed: int,
) -> list[tuple[Path, str, str]]:
    if len(samples) <= target_count:
        return samples

    groups: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    for sample in samples:
        _, plate_text, _ = sample
        key = plate_text[0] if plate_text else "UNKNOWN"
        groups[key].append(sample)

    rng = random.Random(seed)
    for key in groups:
        rng.shuffle(groups[key])

    selected: list[tuple[Path, str, str]] = []
    group_keys = sorted(groups.keys())

    while len(selected) < target_count:
        progressed = False
        for key in group_keys:
            if groups[key] and len(selected) < target_count:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break

    return selected


def copy_samples(
    samples: list[tuple[Path, str]],
    output_image_dir: Path,
    output_label_file: Path,
    name_prefix: str,
) -> int:
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_file.parent.mkdir(parents=True, exist_ok=True)

    written_lines: list[str] = []
    copied_count = 0

    for index, (source_image_path, plate_text) in enumerate(samples, start=1):
        if not source_image_path.exists():
            continue

        target_name = f"{name_prefix}_{index:06d}{source_image_path.suffix.lower()}"
        target_path = output_image_dir / target_name
        shutil.copy2(source_image_path, target_path)
        written_lines.append(f"{target_path}\t{plate_text}\n")
        copied_count += 1

    output_label_file.write_text("".join(written_lines), encoding="utf-8")
    return copied_count


def extend_charset(base_charset: list[str], texts: list[str]) -> list[str]:
    charset = list(base_charset)
    seen = set(charset)

    for text in texts:
        for char in text:
            if char not in seen:
                charset.append(char)
                seen.add(char)

    return charset


def main() -> None:
    # Fixed parameters
    original_ocr_dir = PROJECT_ROOT / "ocr"
    output_ocr_dir = PROJECT_ROOT / "ocr_v2"
    cblprd_root = Path(r"C:\Users\10937\Downloads\CBLPRD-330k_v1")
    cblprd_image_root = cblprd_root / "CBLPRD-330k"
    cblprd_train_file = cblprd_root / "train.txt"
    cblprd_val_file = cblprd_root / "val.txt"

    extra_train_count = 50000
    extra_val_count = 5000
    random_seed = 20260706

    if not original_ocr_dir.exists():
        raise FileNotFoundError(f"Original OCR directory not found: {original_ocr_dir}")
    if not cblprd_root.exists():
        raise FileNotFoundError(f"CBLPRD root not found: {cblprd_root}")
    if not cblprd_image_root.exists():
        raise FileNotFoundError(f"CBLPRD image directory not found: {cblprd_image_root}")
    if not cblprd_train_file.exists() or not cblprd_val_file.exists():
        raise FileNotFoundError("CBLPRD train.txt or val.txt not found")

    output_ocr_dir.mkdir(parents=True, exist_ok=True)

    original_charset = read_charset(original_ocr_dir / "charset.txt")

    original_train = read_label_file(original_ocr_dir / "train_label.txt")
    original_val = read_label_file(original_ocr_dir / "val_label.txt")
    original_test = read_label_file(original_ocr_dir / "test_label.txt")

    cblprd_train_all = parse_cblprd_label_file(cblprd_train_file, cblprd_image_root)
    cblprd_val_all = parse_cblprd_label_file(cblprd_val_file, cblprd_image_root)

    cblprd_train_selected = balanced_sample_by_prefix(
        samples=cblprd_train_all,
        target_count=extra_train_count,
        seed=random_seed,
    )
    cblprd_val_selected = balanced_sample_by_prefix(
        samples=cblprd_val_all,
        target_count=extra_val_count,
        seed=random_seed + 1,
    )

    merged_train = list(original_train) + [(image_path, text) for image_path, text, _ in cblprd_train_selected]
    merged_val = list(original_val) + [(image_path, text) for image_path, text, _ in cblprd_val_selected]
    merged_test = list(original_test)

    copied_train = copy_samples(
        samples=merged_train,
        output_image_dir=output_ocr_dir / "images" / "train",
        output_label_file=output_ocr_dir / "train_label.txt",
        name_prefix="train",
    )
    copied_val = copy_samples(
        samples=merged_val,
        output_image_dir=output_ocr_dir / "images" / "val",
        output_label_file=output_ocr_dir / "val_label.txt",
        name_prefix="val",
    )
    copied_test = copy_samples(
        samples=merged_test,
        output_image_dir=output_ocr_dir / "images" / "test",
        output_label_file=output_ocr_dir / "test_label.txt",
        name_prefix="test",
    )

    merged_texts = [text for _, text in merged_train + merged_val + merged_test]
    merged_charset = extend_charset(original_charset, merged_texts)
    (output_ocr_dir / "charset.txt").write_text("\n".join(merged_charset) + "\n", encoding="utf-8")

    summary = {
        "original_train_count": len(original_train),
        "original_val_count": len(original_val),
        "original_test_count": len(original_test),
        "cblprd_train_total": len(cblprd_train_all),
        "cblprd_val_total": len(cblprd_val_all),
        "cblprd_train_selected": len(cblprd_train_selected),
        "cblprd_val_selected": len(cblprd_val_selected),
        "merged_train_count": copied_train,
        "merged_val_count": copied_val,
        "merged_test_count": copied_test,
        "charset_size": len(merged_charset),
        "output_dir": str(output_ocr_dir),
    }
    (output_ocr_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("OCR v2 dataset preparation finished")
    print(f"Output directory: {output_ocr_dir}")
    print(f"Train count: {copied_train}")
    print(f"Val count: {copied_val}")
    print(f"Test count: {copied_test}")
    print(f"Charset size: {len(merged_charset)}")
    print(f"Summary file: {output_ocr_dir / 'prepare_summary.json'}")


if __name__ == "__main__":
    main()
