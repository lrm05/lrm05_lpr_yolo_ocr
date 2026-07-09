"""Build a PC-compatible TensorRT engine from ONNX.

Work order: 人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务

Use this script when Ultralytics direct TensorRT export hangs or when the engine
must be rebuilt on the current PC GPU.
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt


BASE_DIR = Path(__file__).resolve().parent
ONNX_PATH = BASE_DIR / "best.onnx"
ENGINE_PATH = BASE_DIR / "yolo_best_pc.engine"


def main() -> None:
    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"ONNX file not found: {ONNX_PATH}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX: {ONNX_PATH}")
    if not parser.parse(ONNX_PATH.read_bytes()):
        for index in range(parser.num_errors):
            print(parser.get_error(index))
        raise RuntimeError("Failed to parse ONNX model.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    if hasattr(trt.BuilderFlag, "TF32"):
        config.set_flag(trt.BuilderFlag.TF32)
        print("TF32 enabled for supported layers.")

    print("Building TensorRT engine. This may take several minutes...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed.")

    ENGINE_PATH.write_bytes(serialized_engine)
    print(f"Engine saved to: {ENGINE_PATH}")
    print(f"Engine size: {ENGINE_PATH.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
