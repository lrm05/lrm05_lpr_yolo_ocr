"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务。

批量整图车牌检测 + OCR 识别推理脚本。
运行方式：
    python scripts/infer_plate_batch.py
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


def compute_char_accuracy(predicted: str, target: str) -> float:
    if not target:
        return 0.0
    overlap = min(len(predicted), len(target))
    matches = sum(predicted[i] == target[i] for i in range(overlap))
    return matches / len(target)


def main() -> None:
    # ==================== 固定参数区 ====================
    image_dir = PROJECT_ROOT / "ccpd_samples"
    detector_weights = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    recognizer_weights = PROJECT_ROOT / "runs_ocr_v2" / "best.pt"
    output_dir = PROJECT_ROOT / "infer_batch_results"
    image_output_dir = output_dir / "images"
    summary_csv = output_dir / "batch_results.csv"
    summary_json = output_dir / "batch_summary.json"
    conf = 0.25
    max_images = 50
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ==================================================

    output_dir.mkdir(parents=True, exist_ok=True)
    image_output_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLO(str(detector_weights))
    recognizer, charset = load_recognizer(recognizer_weights, device=device)

    image_paths = sorted(image_dir.glob("*.jpg"))[:max_images]
    if not image_paths:
        raise FileNotFoundError(f"未在目录中找到图片: {image_dir}")

    rows: list[dict[str, object]] = []
    exact_match_count = 0
    char_acc_sum = 0.0
    detected_count = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        sample = parse_ccpd_path(image_path)
        expected_text = sample.plate_text if sample is not None else ""

        results = detector.predict(source=str(image_path), conf=conf, device=device, verbose=False)
        predicted_text = ""
        score = 0.0
        bbox = None

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

            with torch.no_grad():
                crop_tensor = preprocess_crop(crop).to(device)
                logits = recognizer(crop_tensor)
                predicted_text = greedy_decode(logits, charset)[0]

            bbox = crop_bbox
            cv2.rectangle(image, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{predicted_text} {score:.2f}",
                (crop_x1, max(20, crop_y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        exact_match = predicted_text == expected_text and bool(expected_text)
        char_acc = compute_char_accuracy(predicted_text, expected_text) if expected_text else 0.0

        if exact_match:
            exact_match_count += 1
        char_acc_sum += char_acc

        output_image_path = image_output_dir / image_path.name
        cv2.imwrite(str(output_image_path), image)

        rows.append(
            {
                "image": image_path.name,
                "expected_text": expected_text,
                "predicted_text": predicted_text,
                "detected": bool(predicted_text),
                "score": round(score, 4),
                "exact_match": exact_match,
                "char_accuracy": round(char_acc, 4),
                "bbox": list(bbox) if bbox else [],
                "output_image": str(output_image_path),
            }
        )

        print(
            f"{image_path.name} | expected={expected_text} | "
            f"predicted={predicted_text or 'None'} | score={score:.4f}"
        )

    with summary_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "expected_text",
                "predicted_text",
                "detected",
                "score",
                "exact_match",
                "char_accuracy",
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
        "exact_match_count": exact_match_count,
        "exact_match_rate": round(exact_match_count / len(rows), 4) if rows else 0.0,
        "average_char_accuracy": round(char_acc_sum / len(rows), 4) if rows else 0.0,
        "summary_csv": str(summary_csv),
        "output_dir": str(output_dir),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n批量测试完成")
    print(f"总图片数: {summary['total_images']}")
    print(f"检测成功数: {summary['detected_images']}")
    print(f"检测率: {summary['detection_rate']}")
    print(f"整牌准确率: {summary['exact_match_rate']}")
    print(f"平均字符准确率: {summary['average_char_accuracy']}")
    print(f"结果表格: {summary_csv}")
    print(f"结果汇总: {summary_json}")


if __name__ == "__main__":
    main()
