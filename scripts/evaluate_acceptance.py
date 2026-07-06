"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务。

项目验收评测脚本：
1. 基于 CCPD 文件名中的真值车牌框和车牌字符，评估检测定位与 OCR 识别效果。
2. 统计车牌定位相对误差、IoU、整牌识别率、字符识别率和端到端耗时。
3. 生成可直接用于验收说明的 JSON 与 CSV 结果文件。

运行方式：
    python scripts/evaluate_acceptance.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
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
    matches = sum(predicted[index] == target[index] for index in range(overlap))
    return matches / len(target)


def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_relative_bbox_error(predicted: tuple[int, int, int, int], target: tuple[int, int, int, int]) -> float:
    tx1, ty1, tx2, ty2 = target
    px1, py1, px2, py2 = predicted

    target_width = max(1, tx2 - tx1)
    target_height = max(1, ty2 - ty1)

    error_x1 = abs(px1 - tx1) / target_width
    error_y1 = abs(py1 - ty1) / target_height
    error_x2 = abs(px2 - tx2) / target_width
    error_y2 = abs(py2 - ty2) / target_height
    return (error_x1 + error_y1 + error_x2 + error_y2) / 4.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    left_index = int(position)
    right_index = min(left_index + 1, len(ordered) - 1)
    fraction = position - left_index
    return ordered[left_index] * (1.0 - fraction) + ordered[right_index] * fraction


def summarize_scene(rows: list[dict[str, object]], scene_name: str) -> dict[str, object]:
    scene_rows = [row for row in rows if row["scene_proxy"] == scene_name]
    if not scene_rows:
        return {"images": 0, "exact_match_rate": None, "average_char_accuracy": None}

    exact_match_count = sum(1 for row in scene_rows if row["exact_match"])
    average_char_accuracy = sum(float(row["char_accuracy"]) for row in scene_rows) / len(scene_rows)
    return {
        "images": len(scene_rows),
        "exact_match_rate": round(exact_match_count / len(scene_rows), 4),
        "average_char_accuracy": round(average_char_accuracy, 4),
    }


def main() -> None:
    image_dir = PROJECT_ROOT / "data" / "images" / "test"
    detector_weights = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    recognizer_weights = PROJECT_ROOT / "runs_ocr_v2" / "best.pt"
    output_dir = PROJECT_ROOT / "acceptance_results"
    details_csv = output_dir / "acceptance_details.csv"
    summary_json = output_dir / "acceptance_summary.json"
    conf = 0.25
    localization_error_threshold = 0.10
    response_time_threshold_ms = 1000.0
    brightness_night_threshold = 110
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"未在目录中找到测试图片: {image_dir}")

    detector = YOLO(str(detector_weights))
    recognizer, charset = load_recognizer(recognizer_weights, device=device)

    warmup_image = cv2.imread(str(image_paths[0]))
    if warmup_image is not None:
        detector.predict(source=warmup_image, conf=conf, device=device, verbose=False)

    rows: list[dict[str, object]] = []
    detected_count = 0
    exact_match_count = 0
    char_accuracy_sum = 0.0
    iou_values: list[float] = []
    bbox_error_values: list[float] = []
    detect_times_ms: list[float] = []
    ocr_times_ms: list[float] = []
    total_times_ms: list[float] = []
    bbox_error_pass_count = 0
    response_within_1s_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        sample = parse_ccpd_path(image_path)
        if sample is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        expected_text = sample.plate_text
        gt_bbox = sample.bbox
        scene_proxy = "night_proxy" if sample.brightness <= brightness_night_threshold else "day_proxy"

        total_start = time.perf_counter()

        detect_start = time.perf_counter()
        results = detector.predict(source=image, conf=conf, device=device, verbose=False)
        detect_time_ms = (time.perf_counter() - detect_start) * 1000.0

        predicted_text = ""
        score = 0.0
        predicted_bbox: tuple[int, int, int, int] | None = None
        iou = 0.0
        bbox_relative_error = None
        ocr_time_ms = 0.0

        if results and len(results[0].boxes) > 0:
            detected_count += 1
            result = results[0]
            box = result.boxes.xyxy[0].cpu().numpy().tolist()
            score = float(result.boxes.conf[0].cpu().item())
            predicted_bbox = tuple(int(value) for value in box)

            iou = compute_iou(predicted_bbox, gt_bbox)
            bbox_relative_error = compute_relative_bbox_error(predicted_bbox, gt_bbox)
            iou_values.append(iou)
            bbox_error_values.append(bbox_relative_error)
            if bbox_relative_error <= localization_error_threshold:
                bbox_error_pass_count += 1

            height, width = image.shape[:2]
            crop_bbox = expand_bbox(predicted_bbox, width=width, height=height, margin_ratio=0.05)
            crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

            ocr_start = time.perf_counter()
            with torch.no_grad():
                crop_tensor = preprocess_crop(crop).to(device)
                logits = recognizer(crop_tensor)
                predicted_text = greedy_decode(logits, charset)[0]
            ocr_time_ms = (time.perf_counter() - ocr_start) * 1000.0

        total_time_ms = (time.perf_counter() - total_start) * 1000.0

        exact_match = predicted_text == expected_text
        char_accuracy = compute_char_accuracy(predicted_text, expected_text)
        if exact_match:
            exact_match_count += 1
        char_accuracy_sum += char_accuracy

        detect_times_ms.append(detect_time_ms)
        ocr_times_ms.append(ocr_time_ms)
        total_times_ms.append(total_time_ms)
        if total_time_ms <= response_time_threshold_ms:
            response_within_1s_count += 1

        rows.append(
            {
                "image": image_path.name,
                "expected_text": expected_text,
                "predicted_text": predicted_text,
                "detected": bool(predicted_bbox),
                "score": round(score, 4),
                "gt_bbox": list(gt_bbox),
                "pred_bbox": list(predicted_bbox) if predicted_bbox else [],
                "bbox_iou": round(iou, 4),
                "bbox_relative_error": round(float(bbox_relative_error), 4) if bbox_relative_error is not None else "",
                "bbox_error_le_10pct": bool(
                    bbox_relative_error is not None and bbox_relative_error <= localization_error_threshold
                ),
                "exact_match": exact_match,
                "char_accuracy": round(char_accuracy, 4),
                "brightness": sample.brightness,
                "scene_proxy": scene_proxy,
                "detect_time_ms": round(detect_time_ms, 2),
                "ocr_time_ms": round(ocr_time_ms, 2),
                "total_time_ms": round(total_time_ms, 2),
            }
        )

        if index % 200 == 0:
            print(f"已评测 {index}/{len(image_paths)} 张图片")

    with details_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "expected_text",
                "predicted_text",
                "detected",
                "score",
                "gt_bbox",
                "pred_bbox",
                "bbox_iou",
                "bbox_relative_error",
                "bbox_error_le_10pct",
                "exact_match",
                "char_accuracy",
                "brightness",
                "scene_proxy",
                "detect_time_ms",
                "ocr_time_ms",
                "total_time_ms",
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
        "average_char_accuracy": round(char_accuracy_sum / len(rows), 4) if rows else 0.0,
        "mean_bbox_iou": round(sum(iou_values) / len(iou_values), 4) if iou_values else 0.0,
        "median_bbox_iou": round(percentile(iou_values, 0.5), 4) if iou_values else 0.0,
        "mean_bbox_relative_error": round(sum(bbox_error_values) / len(bbox_error_values), 4)
        if bbox_error_values
        else 0.0,
        "bbox_error_le_10pct_count": bbox_error_pass_count,
        "bbox_error_le_10pct_rate": round(bbox_error_pass_count / len(iou_values), 4) if iou_values else 0.0,
        "average_detect_time_ms": round(sum(detect_times_ms) / len(detect_times_ms), 2) if detect_times_ms else 0.0,
        "average_ocr_time_ms": round(sum(ocr_times_ms) / len(ocr_times_ms), 2) if ocr_times_ms else 0.0,
        "average_total_time_ms": round(sum(total_times_ms) / len(total_times_ms), 2) if total_times_ms else 0.0,
        "p95_total_time_ms": round(percentile(total_times_ms, 0.95), 2) if total_times_ms else 0.0,
        "response_within_1s_count": response_within_1s_count,
        "response_within_1s_rate": round(response_within_1s_count / len(total_times_ms), 4) if total_times_ms else 0.0,
        "scene_proxy_metrics": {
            "day_proxy": summarize_scene(rows, "day_proxy"),
            "night_proxy": summarize_scene(rows, "night_proxy"),
        },
        "acceptance_status": {
            "localization_metric_ready": True,
            "recognition_metric_ready": True,
            "response_time_metric_ready": True,
            "strict_day_night_acceptance_ready": False,
            "strict_special_plate_acceptance_ready": False,
            "strict_lprnet_same_split_comparison_ready": False,
        },
        "notes": [
            "车牌定位误差使用 CCPD 真值框与预测框的相对边界误差统计。",
            "day_proxy / night_proxy 仅基于 CCPD 文件名中的 brightness 字段做近似分组，不等于正式白天/夜间标注验收。",
            "如需严格验收白天 >=95% / 夜间 >=90%，仍需补充明确的白天/夜间标注测试集。",
            "如需严格验收特种车牌、军牌、武警牌、临时牌，仍需补充对应专项测试数据。",
        ],
        "details_csv": str(details_csv),
        "output_dir": str(output_dir),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n验收评测完成")
    print(f"总图片数: {summary['total_images']}")
    print(f"检测成功率: {summary['detection_rate']}")
    print(f"整牌识别准确率: {summary['exact_match_rate']}")
    print(f"平均字符准确率: {summary['average_char_accuracy']}")
    print(f"平均定位相对误差: {summary['mean_bbox_relative_error']}")
    print(f"定位误差<=10%占比: {summary['bbox_error_le_10pct_rate']}")
    print(f"平均总耗时(ms): {summary['average_total_time_ms']}")
    print(f"P95总耗时(ms): {summary['p95_total_time_ms']}")
    print(f"1秒内完成占比: {summary['response_within_1s_rate']}")
    print(f"详情文件: {details_csv}")
    print(f"汇总文件: {summary_json}")


if __name__ == "__main__":
    main()
