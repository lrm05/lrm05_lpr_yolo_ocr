"""人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务
这个脚本用于对比 YOLO11 `best.pt`、FP16 TensorRT、INT8 TensorRT 的推理速度。
"""

from pathlib import Path
import time

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent

# 按当前脚本所在目录组织模型文件，避免在不同工作目录下运行时报路径错误。
MODELS = {
    "pt_model": BASE_DIR / "best.pt",
    "trt_fp16": BASE_DIR / "yolo_best_fp16.engine",
    "trt_int8": BASE_DIR / "yolo_best_int8.engine",
}

IMAGE_PATH = BASE_DIR / "02-92_88-268&513_509&595-507&596_264&588_269&514_512&522-0_0_5_26_31_31_19-100-45.jpg"
WARMUP = 10
RUNS = 100


def benchmark_model(model_name: str, model_path: Path) -> None:
    print("\n======================")
    print(f"model: {model_name}")
    print(f"path : {model_path}")

    if not model_path.exists():
        print("skip : model file not found")
        return

    try:
        model = YOLO(str(model_path), task="detect")
    except Exception as exc:
        print(f"skip : failed to load model -> {exc}")
        return

    try:
        for _ in range(WARMUP):
            model(str(IMAGE_PATH), imgsz=640, verbose=False)

        start = time.perf_counter()

        for _ in range(RUNS):
            results = model(str(IMAGE_PATH), imgsz=640, verbose=False)

        end = time.perf_counter()
    except Exception as exc:
        print(f"skip : inference failed -> {exc}")
        return

    total = end - start
    avg = total / RUNS
    fps = 1.0 / avg if avg > 0 else 0.0

    print(f"avg time: {avg * 1000:.3f} ms")
    print(f"fps     : {fps:.2f}")

    output_path = BASE_DIR / f"{model_name}.jpg"
    results[0].save(filename=str(output_path))
    print(f"saved   : {output_path}")


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"test image not found: {IMAGE_PATH}")

    for model_name, model_path in MODELS.items():
        benchmark_model(model_name, model_path)

    print("\nbenchmark finished")
    print("note: TensorRT .engine files must be generated on a compatible GPU/TensorRT environment.")


if __name__ == "__main__":
    main()
