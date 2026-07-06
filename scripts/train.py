"""工单编号：人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务.

YOLOv11 车牌检测训练入口。
本项目按需求固定使用 yolo11n.pt 作为训练起点。
运行方式：
    python train.py
"""

from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolo11n.pt")

    model.train(
        data="yolo_ccpd.yaml",  # 数据集配置文件路径（包含训练/验证/测试集的图片路径和类别信息）
        epochs=100,             # 训练轮数：完整遍历整个训练集的次数
        imgsz=640,              # 输入图片尺寸：将所有训练图片统一缩放到640x640像素
        batch=96,               # 批次大小：每次迭代同时处理96张图片（需要足够显存）
        device=0,               # 训练设备：0表示使用第一张GPU，或使用"cpu"、"0,1"多卡等
        workers=8,              # 数据加载线程数：并行加载数据的进程数，加速数据预处理
        cache=True,             # 是否缓存图片到内存：加速训练，但需要足够RAM（True表示缓存所有图片）
    )


if __name__ == "__main__":
    main()
