"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务。
对比 OCR v1 和 OCR v2 在同一批图片上的识别效果。

运行方式：
    python scripts/compare_ocr_versions.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ccpd_utils import expand_bbox, parse_ccpd_path
from src.crnn import CrnnRecognizer
from src.ocr_utils import Charset, greedy_decode


def load_recognizer(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    charset = Charset(characters=checkpoint["charset"])
    model = CrnnRecognizer(num_classes=charset.num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, charset


def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (168, 48), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor


def load_label_map(label_file: Path) -> dict[str, str]:
    if not label_file.exists():
        return {}

    label_map: dict[str, str] = {}
    for raw_line in label_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        image_name, plate_text = parts
        label_map[image_name.strip()] = plate_text.strip()
    return label_map


def compute_char_accuracy(predicted: str, target: str) -> float:
    if not target:
        return 0.0
    overlap = min(len(predicted), len(target))
    matches = sum(predicted[i] == target[i] for i in range(overlap))
    return matches / len(target)


def predict_text(recognizer, charset, crop_tensor, device: str) -> str:
    with torch.no_grad():
        logits = recognizer(crop_tensor.to(device))
        return greedy_decode(logits, charset)[0]


def main() -> None:
    # Fixed parameters
    image_dir = PROJECT_ROOT / "custom_eval" / "images"
    label_file = PROJECT_ROOT / "custom_eval" / "labels.txt"
    fallback_image_dir = PROJECT_ROOT / "data" / "images" / "test"
    detector_weights = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    recognizer_v1_weights = PROJECT_ROOT / "runs_ocr" / "best.pt"
    recognizer_v2_weights = PROJECT_ROOT / "runs_ocr_v2" / "best.pt"
    output_dir = PROJECT_ROOT / "compare_ocr_results"
    image_output_dir = output_dir / "images"
    summary_csv = output_dir / "compare_results.csv"
    summary_json = output_dir / "compare_summary.json"
    conf = 0.25
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not image_dir.exists():
        image_dir = fallback_image_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    supported_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in supported_suffixes]
    if not image_paths:
        raise FileNotFoundError(f"测试目录中没有可用图片：{image_dir}")

    label_map = load_label_map(label_file)

    detector = YOLO(str(detector_weights))
    recognizer_v1, charset_v1 = load_recognizer(recognizer_v1_weights, device=device)
    recognizer_v2, charset_v2 = load_recognizer(recognizer_v2_weights, device=device)

    rows: list[dict[str, object]] = []
    labeled_count = 0
    detected_count = 0
    v1_exact_count = 0
    v2_exact_count = 0
    v1_char_acc_sum = 0.0
    v2_char_acc_sum = 0.0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        expected_text = label_map.get(image_path.name, "")
        if not expected_text:
            sample = parse_ccpd_path(image_path)
            if sample is not None:
                expected_text = sample.plate_text
        predicted_v1 = ""
        predicted_v2 = ""
        score = 0.0
        bbox = None

        results = detector.predict(source=str(image_path), conf=conf, device=device, verbose=False)
        if results and len(results[0].boxes) > 0:
            detected_count += 1
            result = results[0]
            box = result.boxes.xyxy[0].cpu().numpy().tolist()
            score = float(result.boxes.conf[0].cpu().item())
            x1, y1, x2, y2 = [int(value) for value in box]

            height, width = image.shape[:2]
            crop_bbox = expand_bbox((x1, y1, x2, y2), width=width, height=height, margin_ratio=0.05)
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_tensor = preprocess_crop(crop)

            predicted_v1 = predict_text(recognizer_v1, charset_v1, crop_tensor, device=device)
            predicted_v2 = predict_text(recognizer_v2, charset_v2, crop_tensor, device=device)

            bbox = crop_bbox
            cv2.rectangle(image, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"v1:{predicted_v1}",
                (crop_x1, max(20, crop_y1 - 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )
            cv2.putText(
                image,
                f"v2:{predicted_v2}",
                (crop_x1, max(40, crop_y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        has_label = bool(expected_text)
        v1_exact = predicted_v1 == expected_text and has_label
        v2_exact = predicted_v2 == expected_text and has_label
        v1_char_acc = compute_char_accuracy(predicted_v1, expected_text) if has_label else None
        v2_char_acc = compute_char_accuracy(predicted_v2, expected_text) if has_label else None

        if has_label:
            labeled_count += 1
            if v1_exact:
                v1_exact_count += 1
            if v2_exact:
                v2_exact_count += 1
            v1_char_acc_sum += float(v1_char_acc)
            v2_char_acc_sum += float(v2_char_acc)

        if has_label:
            if float(v2_char_acc) > float(v1_char_acc):
                better_model = "v2"
            elif float(v2_char_acc) < float(v1_char_acc):
                better_model = "v1"
            else:
                better_model = "same"
        else:
            better_model = ""

        output_image_path = image_output_dir / image_path.name
        cv2.imwrite(str(output_image_path), image)

        rows.append(
            {
                "image": image_path.name,
                "expected_text": expected_text,
                "predicted_v1": predicted_v1,
                "predicted_v2": predicted_v2,
                "detected": bool(predicted_v1 or predicted_v2),
                "score": round(score, 4),
                "v1_exact_match": v1_exact if has_label else "",
                "v2_exact_match": v2_exact if has_label else "",
                "v1_char_accuracy": round(float(v1_char_acc), 4) if v1_char_acc is not None else "",
                "v2_char_accuracy": round(float(v2_char_acc), 4) if v2_char_acc is not None else "",
                "better_model": better_model,
                "bbox": list(bbox) if bbox else [],
                "output_image": str(output_image_path),
            }
        )

        print(
            f"{image_path.name} | "
            f"expected={expected_text or 'None'} | "
            f"v1={predicted_v1 or 'None'} | "
            f"v2={predicted_v2 or 'None'}"
        )

    with summary_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "expected_text",
                "predicted_v1",
                "predicted_v2",
                "detected",
                "score",
                "v1_exact_match",
                "v2_exact_match",
                "v1_char_accuracy",
                "v2_char_accuracy",
                "better_model",
                "bbox",
                "output_image",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_images": len(rows),
        "detected_images": detected_count,
        "detection_rate": round(detected_count / len(rows), 4) if rows else 0.0,
        "labeled_images": labeled_count,
        "v1_exact_match_count": v1_exact_count if labeled_count else None,
        "v2_exact_match_count": v2_exact_count if labeled_count else None,
        "v1_exact_match_rate": round(v1_exact_count / labeled_count, 4) if labeled_count else None,
        "v2_exact_match_rate": round(v2_exact_count / labeled_count, 4) if labeled_count else None,
        "v1_average_char_accuracy": round(v1_char_acc_sum / labeled_count, 4) if labeled_count else None,
        "v2_average_char_accuracy": round(v2_char_acc_sum / labeled_count, 4) if labeled_count else None,
        "summary_csv": str(summary_csv),
        "output_dir": str(output_dir),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nOCR v1/v2 对比测试完成")
    print(f"总图片数: {summary['total_images']}")
    print(f"检测率: {summary['detection_rate']}")
    print(f"带标签图片数: {summary['labeled_images']}")
    print(f"v1 整牌准确率: {summary['v1_exact_match_rate']}")
    print(f"v2 整牌准确率: {summary['v2_exact_match_rate']}")
    print(f"v1 平均字符准确率: {summary['v1_average_char_accuracy']}")
    print(f"v2 平均字符准确率: {summary['v2_average_char_accuracy']}")
    print(f"测试图片目录: {image_dir}")
    print(f"结果表格: {summary_csv}")
    print(f"结果汇总: {summary_json}")


if __name__ == "__main__":
    main()
