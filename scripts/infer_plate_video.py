"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务。

文件作用：这个文件用于视频车牌检测与 OCR 识别推理。

视频推理增强版：
1. 使用 YOLOv11 在视频帧中检测车牌。
2. 对车牌裁剪图做边距外扩、对比度增强和锐化，减少视频模糊对 OCR 的影响。
3. 使用 CRNN + CTC 完成字符识别。
4. 使用多帧投票融合，优先显示连续多帧更稳定的车牌结果。
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ccpd_utils import expand_bbox
from src.crnn import CrnnRecognizer
from src.ocr_utils import Charset


@dataclass
class PlateTrack:
    """用一个轻量级轨迹保存同一辆车在多帧中的 OCR 投票结果。"""

    track_id: int
    bbox: tuple[int, int, int, int]
    last_seen: int
    votes: Counter[str] = field(default_factory=Counter)
    best_score: dict[str, float] = field(default_factory=dict)

    def update(self, bbox: tuple[int, int, int, int], frame_index: int, text: str, score: float) -> None:
        self.bbox = bbox
        self.last_seen = frame_index
        if is_valid_plate_text(text):
            self.votes[text] += max(score, 0.1)
            self.best_score[text] = max(self.best_score.get(text, 0.0), score)

    def best_text(self, fallback: str = "") -> str:
        if not self.votes:
            return fallback
        return self.votes.most_common(1)[0][0]


def load_recognizer(checkpoint_path: Path, device: str):
    # OCR 权重中保存了字符集和模型参数，视频推理时需要同步恢复。
    checkpoint = torch.load(checkpoint_path, map_location=device)
    charset = Charset(characters=checkpoint["charset"])
    model = CrnnRecognizer(num_classes=charset.num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, charset


def enhance_plate_crop(crop: np.ndarray) -> np.ndarray:
    """对视频车牌裁剪图做轻量增强，让 OCR 看到更清晰的字符边缘。"""

    if crop.size == 0:
        return crop

    height, width = crop.shape[:2]
    if width < 100:
        scale = max(2.0, 120.0 / max(width, 1))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return gray


def preprocess_crop(crop):
    # 视频帧里的车牌经常比较小或有压缩噪声，因此这里先增强再 resize 到 OCR 输入尺寸。
    gray = enhance_plate_crop(crop)
    if gray.size == 0:
        return None
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (168, 48), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor


def decode_with_confidence(logits: torch.Tensor, charset: Charset) -> tuple[str, float]:
    """CTC 贪心解码，同时估计识别置信度，便于过滤很不稳定的视频帧。"""

    probabilities = torch.softmax(logits, dim=2)
    prediction = probabilities.argmax(dim=2).transpose(0, 1)[0]
    max_probs = probabilities.max(dim=2).values.transpose(0, 1)[0]

    deduplicated: list[int] = []
    selected_probs: list[float] = []
    previous = -1
    for value, prob in zip(prediction.tolist(), max_probs.tolist()):
        if value != previous and value != 0:
            deduplicated.append(value)
            selected_probs.append(float(prob))
        previous = value

    text = charset.decode(deduplicated)
    confidence = sum(selected_probs) / len(selected_probs) if selected_probs else 0.0
    return text, confidence


def is_valid_plate_text(text: str) -> bool:
    """过滤明显不合理的 OCR 输出，避免短字符或乱码参与多帧投票。"""

    if not text:
        return False
    return 6 <= len(text) <= 8


def recognize_crop(recognizer, charset: Charset, crop, device: str) -> tuple[str, float]:
    # 如果车牌裁剪图太小，直接识别通常容易出错，因此先过滤掉质量明显不足的裁剪。
    if crop.size == 0:
        return "", 0.0
    crop_height, crop_width = crop.shape[:2]
    if crop_width < 24 or crop_height < 8:
        return "", 0.0

    with torch.no_grad():
        crop_tensor = preprocess_crop(crop)
        if crop_tensor is None:
            return "", 0.0
        crop_tensor = crop_tensor.to(device)
        logits = recognizer(crop_tensor)
        return decode_with_confidence(logits, charset)


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def match_track(
    tracks: list[PlateTrack],
    bbox: tuple[int, int, int, int],
    frame_index: int,
    max_missing_frames: int,
) -> PlateTrack | None:
    """按中心点距离匹配最近轨迹，用于把同一车牌的多帧识别结果合并。"""

    cx, cy = bbox_center(bbox)
    best_track = None
    best_distance = float("inf")
    box_width = max(bbox[2] - bbox[0], 1)

    for track in tracks:
        if frame_index - track.last_seen > max_missing_frames:
            continue
        tx, ty = bbox_center(track.bbox)
        distance = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
        distance_threshold = max(80.0, box_width * 3.0)
        if distance < distance_threshold and distance < best_distance:
            best_track = track
            best_distance = distance

    return best_track


def main() -> None:
    input_video = PROJECT_ROOT / "1.mp4"
    output_video = PROJECT_ROOT / "video_infer_result.mp4"
    detector_weights = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
    recognizer_weights = PROJECT_ROOT / "runs_ocr_v2" / "best.pt"
    # Dashcam plates are much smaller than the cropped CCPD-style training images,
    # so use a larger inference size and start from the segment with visible plates.
    # 视频车牌较小，检测阈值不能太高；后续通过 OCR 置信度和多帧投票提高最终稳定性。
    conf = 0.08
    imgsz = 1280
    start_frame = 2000
    max_frames = 300
    crop_margin_ratio = 0.10
    ocr_min_confidence = 0.35
    max_missing_frames = 20
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not input_video.exists():
        raise FileNotFoundError(f"Video not found: {input_video}")

    detector = YOLO(str(detector_weights))
    recognizer, charset = load_recognizer(recognizer_weights, device=device)

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    remaining_frames = max(total_frames - start_frame, 0) if total_frames > 0 else max_frames
    frames_to_process = min(remaining_frames, max_frames) if remaining_frames > 0 else max_frames

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_video}")

    processed_frames = 0
    detected_frames = 0
    next_track_id = 1
    tracks: list[PlateTrack] = []
    recognition_counter: Counter[str] = Counter()
    start_time = time.perf_counter()

    while processed_frames < frames_to_process:
        ok, frame = capture.read()
        if not ok:
            break

        results = detector.predict(source=frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
        if results and len(results[0].boxes) > 0:
            detected_frames += 1
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = [int(value) for value in box.tolist()]
                crop_x1, crop_y1, crop_x2, crop_y2 = expand_bbox(
                    (x1, y1, x2, y2),
                    width=width,
                    height=height,
                    margin_ratio=crop_margin_ratio,
                )
                crop_bbox = (crop_x1, crop_y1, crop_x2, crop_y2)
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                raw_text, ocr_confidence = recognize_crop(recognizer, charset, crop, device=device)

                track = match_track(tracks, crop_bbox, processed_frames, max_missing_frames)
                if track is None:
                    track = PlateTrack(track_id=next_track_id, bbox=crop_bbox, last_seen=processed_frames)
                    tracks.append(track)
                    next_track_id += 1

                if ocr_confidence >= ocr_min_confidence:
                    track.update(crop_bbox, processed_frames, raw_text, ocr_confidence)
                    if is_valid_plate_text(raw_text):
                        recognition_counter[raw_text] += 1
                else:
                    track.update(crop_bbox, processed_frames, "", 0.0)

                text = track.best_text(fallback=raw_text if is_valid_plate_text(raw_text) else "")
                label = f"{text} det:{score:.2f} ocr:{ocr_confidence:.2f}" if text else f"det:{score:.2f}"

                cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (crop_x1, max(25, crop_y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        tracks = [track for track in tracks if processed_frames - track.last_seen <= max_missing_frames]

        writer.write(frame)
        processed_frames += 1
        if processed_frames % 50 == 0:
            print(f"processed {processed_frames}/{frames_to_process} frames")

    elapsed = time.perf_counter() - start_time
    capture.release()
    writer.release()

    avg_time_ms = elapsed / processed_frames * 1000 if processed_frames else 0.0
    print(f"input_video: {input_video}")
    print(f"output_video: {output_video}")
    print(f"start_frame: {start_frame}")
    print(f"processed_frames: {processed_frames}")
    print(f"detected_frames: {detected_frames}")
    print(f"avg_time_ms_per_frame: {avg_time_ms:.2f}")
    if recognition_counter:
        print("top_recognition_results:")
        for text, count in recognition_counter.most_common(5):
            print(f"  {text}: {count}")


if __name__ == "__main__":
    main()
