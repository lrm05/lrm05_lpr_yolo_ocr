# 智能交通路口车牌识别升级项目

## 1. 项目简介
本项目面向智能交通路口管理场景，完成车牌识别算法的迭代升级。整体方案采用 `YOLOv11 + OCR` 两阶段流程：

- 第 1 阶段：使用 `YOLOv11` 完成车辆图像中的车牌检测与定位
- 第 2 阶段：使用 `OCR(CRNN + CTC)` 完成车牌字符识别
- 对比实验：补充完成 `LPRNet` 与 `PaddleOCR` 的同任务识别对比

项目目标是提升交通路口场景下的车牌定位精度、字符识别精度与整体响应速度，并形成可复现的训练、推理、评测与验收流程。

代码注释中已包含工单编号：`人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务`。

## 2. 任务目标
本项目围绕以下需求实现：

- 车牌定位误差控制在 `10%` 以内
- 日间车牌字符识别准确率不低于 `95%`
- 夜间车牌字符识别准确率不低于 `90%`
- 从检测到输出识别结果的总响应时间不超过 `1 秒`
- 技术路线采用 `YOLOv11 车牌检测 + OCR 文字识别`

## 3. 技术路线
整体识别流程如下：

1. 输入车辆图像
2. 使用 `YOLOv11` 检测车牌位置
3. 根据检测框裁剪车牌区域
4. 将车牌裁剪图送入 OCR 识别模型
5. 输出完整车牌字符串
6. 统计检测、识别与响应时间指标

当前项目主要采用以下三条识别路线进行对比：

- 主方案：`OCR(CRNN + CTC)`
- 基线方案：`LPRNet`
- 第三方对照方案：`PaddleOCR`

## 4. 当前结果摘要
### 4.1 验收结果摘要
来自 `acceptance_results/acceptance_summary.json` 的结果如下：

- 测试集总数：`3000`
- 车牌检测成功率：`1.0000`
- 整牌识别准确率：`0.9933`
- 平均字符准确率：`0.9988`
- 平均定位相对误差：`0.0344`
- 定位误差 `<= 10%` 占比：`0.9930`
- 平均总耗时：`39.09 ms`
- `P95` 总耗时：`49.95 ms`
- `1 秒内完成` 占比：`1.0000`

### 4.2 识别模型对比摘要
当前项目内三种识别方案的测试结果摘要如下：

- `OCR(CRNN + CTC)`：`exact_accuracy = 0.9923`
- `LPRNet`：`exact_accuracy = 0.9813`
- `PaddleOCR`：`acc = 0.9480`

当前项目中，`OCR(CRNN + CTC)` 是识别效果最好的方案，因此作为最终推荐识别模型。

## 5. 运行环境
### 5.1 建议环境
- 操作系统：`Windows 10/11` 或 `Linux`
- Python：`3.10`
- CUDA：有 NVIDIA GPU 时建议开启
- PyTorch：建议安装支持 CUDA 的版本

### 5.2 主要依赖
建议至少安装以下依赖：

```bash
pip install torch torchvision torchaudio
pip install ultralytics opencv-python numpy pandas matplotlib pillow tqdm pyyaml
```

如果需要运行 PaddleOCR 对比实验，还需要额外准备：

```bash
pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
pip install -r PaddleOCR-main/requirements.txt
```

说明：

- `YOLOv11` 检测训练与推理主要依赖 `ultralytics`
- 自研 OCR 与 LPRNet 训练主要依赖 `torch`
- PaddleOCR 对比实验依赖 `PaddleOCR-main`

## 6. 项目目录结构
```text
9-Sampling_of_CCPD_files
├─ acceptance_results/          # 项目验收结果，包含汇总指标和逐样本明细
├─ analysis/                    # 数据分析、模型对比分析、验收说明文档
├─ compare_ocr_results/         # OCR v1 与 OCR v2 对比结果
├─ cs/                          # YOLO PT 与 TensorRT 推理测速相关文件
├─ data/                        # YOLO 检测数据集，含 train/val/test 图像与标签
├─ infer_batch_results/         # 批量推理输出结果
├─ ocr/                         # 第一版 OCR 识别数据集
├─ ocr_v2/                      # 增强版 OCR 识别数据集
├─ paddleocr_output/            # PaddleOCR 训练输出结果
├─ paddleocr_rec/               # PaddleOCR 识别训练所需数据
├─ runs/                        # YOLOv11 检测训练结果
├─ runs_lprnet/                 # LPRNet 训练结果
├─ runs_ocr/                    # OCR v1 训练结果
├─ runs_ocr_v2/                 # OCR v2 训练结果
├─ scripts/                     # 项目主要执行脚本入口
├─ src/                         # 模型结构与工具函数源码
├─ PaddleOCR-main/              # PaddleOCR 源码目录
├─ split_manifest.json          # 数据集切分记录
├─ yolo_ccpd.yaml               # YOLOv11 数据配置文件
├─ yolo11n.pt                   # YOLOv11 预训练权重
└─ README.md                    # 项目说明文档
```

## 7. 核心脚本说明
### 7.1 数据准备与分析
- `scripts/analyze_ccpd.py`
  用于分析 CCPD 数据集分布情况，输出 JSON 和 Markdown 摘要。
- `scripts/prepare_datasets.py`
  用于把原始 CCPD 图片整理为 YOLO 检测数据集和 OCR 识别数据集。
- `scripts/prepare_ocr_v2_dataset.py`
  用于生成增强版 OCR 数据集。
- `scripts/prepare_paddleocr_rec_dataset.py`
  用于将现有 OCR 数据转成 PaddleOCR 所需格式。

### 7.2 模型训练
- `scripts/train.py`
  训练 `YOLOv11` 车牌检测模型。
- `scripts/train_ocr.py`
  训练第一版 `OCR(CRNN + CTC)` 识别模型。
- `scripts/train_ocr_v2.py`
  训练增强版 `OCR(CRNN + CTC)` 识别模型。
- `scripts/train_lprnet.py`
  训练 `LPRNet` 基线识别模型。

### 7.3 推理与评测
- `scripts/infer_plate.py`
  对单张图片完成“检测 + 识别”完整推理。
- `scripts/infer_plate_batch.py`
  对多张图片进行批量推理，并输出统计结果。
- `scripts/evaluate_recognizers.py`
  对 `OCR(CRNN + CTC)` 与 `LPRNet` 进行统一测试评估。
- `scripts/evaluate_acceptance.py`
  按项目验收口径输出检测、识别和响应时间结果。
- `scripts/compare_ocr_versions.py`
  对比 OCR v1 和 OCR v2 的识别表现。
- `cs/compare_speed.py`
  对比 `PyTorch` 与 `TensorRT` 引擎的推理速度。

## 8. 数据准备流程
### 8.1 原始数据集分析
在项目根目录下执行：

```bash
python scripts/analyze_ccpd.py --dataset-dir Sampling_of_CCPD_files --output-dir analysis
```

输出内容：

- `analysis/ccpd_summary.json`
- `analysis/ccpd_summary.md`

### 8.2 生成 YOLO 与 OCR 数据集
如果需要从原始 CCPD 图片重新生成训练数据，可以执行：

```bash
python scripts/prepare_datasets.py --dataset-dir Sampling_of_CCPD_files --output-dir .
```

执行后会生成或更新：

- `data/`
- `ocr/`
- `yolo_ccpd.yaml`
- `split_manifest.json`

### 8.3 生成增强版 OCR 数据集
如果需要增强 OCR 数据集，可以执行：

```bash
python scripts/prepare_ocr_v2_dataset.py
```

执行后会生成或更新：

- `ocr_v2/`

## 9. 训练流程
### 9.1 训练 YOLOv11 车牌检测模型
```bash
python scripts/train.py
```

默认输出目录：

- `runs/detect/train/`

核心权重文件：

- `runs/detect/train/weights/best.pt`

### 9.2 训练 OCR v1 模型
```bash
python scripts/train_ocr.py
```

默认输出目录：

- `runs_ocr/`

核心权重文件：

- `runs_ocr/best.pt`

### 9.3 训练 OCR v2 模型
```bash
python scripts/train_ocr_v2.py
```

默认输出目录：

- `runs_ocr_v2/`

核心权重文件：

- `runs_ocr_v2/best.pt`

### 9.4 训练 LPRNet 基线模型
```bash
python scripts/train_lprnet.py
```

默认输出目录：

- `runs_lprnet/`

核心权重文件：

- `runs_lprnet/best.pt`

### 9.5 训练 PaddleOCR 对照模型
PaddleOCR 训练在 `PaddleOCR-main/` 目录中执行，配置文件采用项目内准备好的车牌识别配置。

训练完成后，结果通常输出到：

- `paddleocr_output/plate_rec/`

## 10. 推理与测试
### 10.1 单张图片推理
```bash
python scripts/infer_plate.py
```

默认行为：

- 自动读取测试图片
- 自动加载 `runs/detect/train/weights/best.pt`
- 自动加载 `runs_ocr_v2/best.pt`
- 输出结果图片到 `infer_result.jpg`

### 10.2 批量图片推理
```bash
python scripts/infer_plate_batch.py
```

默认输出目录：

- `infer_batch_results/images/`
- `infer_batch_results/batch_results.csv`
- `infer_batch_results/batch_summary.json`

### 10.3 OCR 与 LPRNet 统一评测
```bash
python scripts/evaluate_recognizers.py
```

默认输出文件：

- `analysis/recognizer_compare_summary.json`

### 10.4 项目验收评测
```bash
python scripts/evaluate_acceptance.py
```

默认输出文件：

- `acceptance_results/acceptance_details.csv`
- `acceptance_results/acceptance_summary.json`

## 11. TensorRT 加速测速
如果已经完成 `best.pt` 到 TensorRT 引擎的导出，可以执行：

```bash
python cs/compare_speed.py
```

该脚本用于对比：

- 原始 `PyTorch .pt` 模型
- `FP16 TensorRT engine`
- `INT8 TensorRT engine`

测速相关文件统一放在：

- `cs/`

## 12. 结果文件说明
项目中最重要的结果文件如下：

- `runs/detect/train/weights/best.pt`
  YOLOv11 车牌检测最优权重。
- `runs_ocr_v2/best.pt`
  当前推荐使用的 OCR 最优识别权重。
- `runs_lprnet/best.pt`
  LPRNet 对照实验最优权重。
- `acceptance_results/acceptance_summary.json`
  项目验收核心指标汇总。
- `analysis/recognizer_compare_summary.json`
  OCR 与 LPRNet 的统一测试对比结果。
- `analysis/OCR、LPRNet与PaddleOCR对比分析.md`
  三种识别路线的对比分析文档。
- `analysis/项目验收说明.md`
  按项目需求整理的验收结论说明。

## 13. GitHub 上传建议
如果要将本项目上传到 GitHub，建议：

### 13.1 建议上传
- `scripts/`
- `src/`
- `analysis/`
- `cs/` 中的脚本与说明文件
- `README.md`
- `yolo_ccpd.yaml`
- 适量示例图片与示例结果


## 14. 当前推荐使用方式
如果只保留一条主流程，推荐如下：

1. 使用 `YOLOv11` 检测模型：`runs/detect/train/weights/best.pt`
2. 使用 OCR 识别模型：`runs_ocr_v2/best.pt`
3. 单张演示运行：`python scripts/infer_plate.py`
4. 批量评测运行：`python scripts/evaluate_acceptance.py`

## 15. 后续可继续优化方向
- 补充更多省份、更多特殊车牌类型的数据
- 补充明确的白天/夜间标注测试集
- 补充雨天、逆光、遮挡、模糊等复杂场景样本
- 对 OCR 识别模型继续做更大规模数据增强
- 对部署端继续推进 TensorRT 加速落地
