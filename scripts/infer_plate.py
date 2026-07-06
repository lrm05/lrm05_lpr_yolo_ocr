"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务。

单张图片车牌检测 + OCR 识别推理脚本。
运行方式：
    python scripts/infer_plate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ccpd_utils import expand_bbox
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


def find_default_image() -> Path:
    image_dir = PROJECT_ROOT / "data" / "images" / "test"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"未在目录中找到测试图片: {image_dir}")
    return image_paths[0]


def main() -> None:
    image_path = find_default_image()
    detector_weights = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    recognizer_weights = PROJECT_ROOT / "runs_ocr_v2" / "best.pt"
    output_path = PROJECT_ROOT / "infer_result.jpg"
    conf = 0.25
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    detector = YOLO(str(detector_weights))
    recognizer, charset = load_recognizer(recognizer_weights, device=device)

    results = detector.predict(source=str(image_path), conf=conf, device=device, verbose=False)
    if not results or len(results[0].boxes) == 0:
        print("未检测到车牌。")
        return

    result = results[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    height, width = image.shape[:2]

    for index, (box, score) in enumerate(zip(boxes, scores), start=1):
        x1, y1, x2, y2 = [int(value) for value in box.tolist()]
        crop_bbox = expand_bbox((x1, y1, x2, y2), width=width, height=height, margin_ratio=0.05)
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]

        with torch.no_grad():
            crop_tensor = preprocess_crop(crop).to(device)
            logits = recognizer(crop_tensor)
            text = greedy_decode(logits, charset)[0]

        cv2.rectangle(image, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{text} {score:.2f}",
            (crop_x1, max(20, crop_y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        print(f"plate_{index}: text={text}, score={score:.4f}, bbox={crop_bbox}")

    cv2.imwrite(str(output_path), image)
    print(f"可视化结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
