"""Export YOLOv11 plate detector to TensorRT engines.

Work order: 人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务

TensorRT engine files are device-specific. Run this script separately on the PC
and on the development board, then benchmark the engine generated on that device.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PT_MODEL = BASE_DIR / "best.pt"
DATA_YAML = PROJECT_ROOT / "yolo_ccpd.yaml"
IMG_SIZE = 640


def export_fp16() -> Path:
    model = YOLO(str(PT_MODEL))
    exported_path = Path(
        model.export(
            format="engine",
            imgsz=IMG_SIZE,
            half=True,
            device=0,
            simplify=True,
        )
    )
    target_path = BASE_DIR / "yolo_best_pc_fp16.engine"
    shutil.move(str(exported_path), target_path)
    return target_path


def export_int8() -> Path:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"INT8 calibration data config not found: {DATA_YAML}")

    model = YOLO(str(PT_MODEL))
    exported_path = Path(
        model.export(
            format="engine",
            imgsz=IMG_SIZE,
            int8=True,
            data=str(DATA_YAML),
            device=0,
            simplify=True,
        )
    )
    target_path = BASE_DIR / "yolo_best_pc_int8.engine"
    shutil.move(str(exported_path), target_path)
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TensorRT engines for the current device.")
    parser.add_argument("--include-int8", action="store_true", help="Also export INT8 engine with calibration data.")
    args = parser.parse_args()

    if not PT_MODEL.exists():
        raise FileNotFoundError(f"PT model not found: {PT_MODEL}")

    print("Exporting FP16 TensorRT engine...")
    fp16_path = export_fp16()
    print(f"FP16 engine saved to: {fp16_path}")

    if args.include_int8:
        print("Exporting INT8 TensorRT engine...")
        int8_path = export_int8()
        print(f"INT8 engine saved to: {int8_path}")
    else:
        print("INT8 export skipped. Run with --include-int8 if calibration export is required.")


if __name__ == "__main__":
    main()
