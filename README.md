# 智能交通路口车牌识别升级项目

本项目用于完成“智能交通路口管理”场景下的车牌识别算法升级任务，采用：
- `YOLOv11` 完成车牌检测
- `OCR(CRNN + CTC)` 完成车牌字符识别

项目目标是提升交通路口场景下的车牌定位精度、字符识别精度和整体响应速度，并为后续多省份、多场景扩展提供基础。

代码注释已包含工单编号：
`人工智能 CV-智能交通路口管理-车辆、行人目标检测与跟踪算法升级任务`

## 一、项目当前结论

当前项目已经完成从 0 到 1 的主体实现，已具备：
- 车牌检测模型训练能力
- OCR 识别模型训练能力
- 单张图片推理
- 批量图片推理
- OCR v1 / OCR v2 对比评测
- 验收指标评测
- 验收文档与分析文档整理

当前主用模型为：
- 检测模型：[runs/detect/train/weights/best.pt](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\runs\detect\train\weights\best.pt)
- OCR 模型：[runs_ocr_v2/best.pt](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\runs_ocr_v2\best.pt)

## 二、当前验收结果摘要

正式验收评测文件：
- [acceptance_summary.json](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\acceptance_results\acceptance_summary.json)
- [acceptance_details.csv](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\acceptance_results\acceptance_details.csv)

当前 3000 张测试图评测结果：
- 车牌检测成功率：`1.0`
- 整牌识别准确率：`0.9933`
- 平均字符准确率：`0.9988`
- 平均定位相对误差：`0.0344`
- 定位误差 `<=10%` 占比：`0.9930`
- 平均总耗时：`39.09 ms`
- `P95` 总耗时：`49.95 ms`
- `1 秒内完成` 占比：`1.0`

OCR v1 / OCR v2 对比结果：
- [compare_summary.json](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\compare_ocr_results\compare_summary.json)

当前结果：
- `v1_exact_match_rate = 0.9920`
- `v2_exact_match_rate = 0.9933`
- `v1_average_char_accuracy = 0.9987`
- `v2_average_char_accuracy = 0.9988`

当前结论：
- `OCR v2` 已略优于 `OCR v1`
- 当前项目后续识别模型默认使用 `runs_ocr_v2/best.pt`

## 三、项目目录结构

```text
Sampling_of_CCPD_files
|___analysis
|   |___ccpd_summary.json
|   |___ccpd_summary.md
|   |___OCR与LPRNet对比分析.md
|   |___项目验收说明.md
|
|___acceptance_results
|   |___acceptance_details.csv
|   |___acceptance_summary.json
|
|___compare_ocr_results
|   |___images
|   |___compare_results.csv
|   |___compare_summary.json
|
|___data
|   |___images
|   |   |___train
|   |   |___val
|   |   |___test
|   |___labels
|       |___train
|       |___val
|       |___test
|
|___infer_batch_results
|   |___images
|   |___batch_results.csv
|   |___batch_summary.json
|
|___ocr
|   |___images
|   |   |___train
|   |   |___val
|   |   |___test
|   |___charset.txt
|   |___train_label.txt
|   |___val_label.txt
|   |___test_label.txt
|
|___ocr_v2
|   |___images
|   |   |___train
|   |   |___val
|   |   |___test
|   |___charset.txt
|   |___prepare_summary.json
|   |___train_label.txt
|   |___val_label.txt
|   |___test_label.txt
|
|___runs
|   |___detect
|       |___train
|           |___weights
|               |___best.pt
|               |___last.pt
|
|___runs_ocr
|   |___best.pt
|   |___metrics.json
|
|___runs_ocr_v2
|   |___best.pt
|   |___metrics.json
|
|___scripts
|   |___analyze_ccpd.py
|   |___prepare_datasets.py
|   |___prepare_ocr_v2_dataset.py
|   |___train.py
|   |___train_ocr.py
|   |___train_ocr_v2.py
|   |___evaluate_acceptance.py
|   |___infer_plate.py
|   |___infer_plate_batch.py
|   |___infer_custom_folder.py
|   |___compare_ocr_versions.py
|
|___src
|   |___ccpd_utils.py
|   |___crnn.py
|   |___ocr_utils.py
|
|___CCPD2019_数据格式说明.md
|___ccpd2019_to_paddleocr_legacy.py
|___infer_result.jpg
|___README.md
|___split_manifest.json
|___yolo11n.pt
|___yolo26n.pt
|___yolo_ccpd.yaml
```

## 四、关键目录说明

### `analysis`

项目分析和验收文档目录。

- [ccpd_summary.md](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\analysis\ccpd_summary.md)
  数据集分析说明。
- [OCR与LPRNet对比分析.md](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\analysis\OCR与LPRNet对比分析.md)
  OCR 路线与 LPRNet 路线分析说明。
- [项目验收说明.md](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\analysis\项目验收说明.md)
  面向交付和验收的正式说明。

### `acceptance_results`

正式验收评测结果目录。

- `acceptance_details.csv`
  每张测试图的定位、识别、耗时明细。
- `acceptance_summary.json`
  验收汇总指标。

### `data`

YOLOv11 检测训练数据目录。

- `data/images/train`
- `data/images/val`
- `data/images/test`
- `data/labels/train`
- `data/labels/val`
- `data/labels/test`

### `ocr`

OCR v1 训练数据目录。

### `ocr_v2`

OCR v2 增强训练数据目录。

### `runs`

YOLOv11 检测训练输出目录。

### `runs_ocr`

OCR v1 训练输出目录。

### `runs_ocr_v2`

OCR v2 训练输出目录，也是当前主用 OCR 模型目录。

### `compare_ocr_results`

OCR v1 / OCR v2 对比结果目录。

### `infer_batch_results`

批量测试结果目录。

### `scripts`

项目主要运行入口目录，你平时主要直接运行这里的脚本。

### `src`

底层源码目录，一般不直接运行，主要被 `scripts/` 中的脚本调用。

## 五、主要脚本说明

- [analyze_ccpd.py](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\scripts\analyze_ccpd.py)
  分析 CCPD 数据集。
- [prepare_datasets.py](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\scripts\prepare_datasets.py)
  生成 YOLO 检测数据和 OCR v1 数据。
- [prepare_ocr_v2_dataset.py](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\scripts\prepare_ocr_v2_dataset.py)
  生成 OCR v2 增强数据。
- [train.py](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\scripts\train.py)
  训练 YOLOv11 检测模型。
- [train_ocr.py](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\scripts\train_ocr.py)
  训练 OCR v1。
- [train_ocr_v2.py](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\scripts\train_ocr_v2.py)
  训练 OCR v2。
- [infer_plate.py](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\scripts\infer_plate.py)
  单张图片推理。
- [infer_plate_batch.py](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\scripts\infer_plate_batch.py)
  批量样本推理。
- [infer_custom_folder.py](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\scripts\infer_custom_folder.py)
  批量自定义图片推理。
- [compare_ocr_versions.py](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\scripts\compare_ocr_versions.py)
  OCR v1 / OCR v2 对比评测。
- [evaluate_acceptance.py](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\scripts\evaluate_acceptance.py)
  验收指标评测。

## 六、常用命令

### 1. 训练 YOLOv11 检测模型

```powershell
python scripts/train.py
```

### 2. 训练 OCR v1

```powershell
python scripts/train_ocr.py
```

### 3. 训练 OCR v2

```powershell
python scripts/train_ocr_v2.py
```

### 4. 单张图片测试

```powershell
python scripts/infer_plate.py
```

### 5. 批量图片测试

```powershell
python scripts/infer_plate_batch.py
```

### 6. OCR v1 / v2 对比测试

```powershell
python scripts/compare_ocr_versions.py
```

### 7. 验收评测

```powershell
python scripts/evaluate_acceptance.py
```

## 七、当前项目状态判断

按“项目主体是否完成”看：
- 已基本完成

按“是否已具备交付基础”看：
- 已具备较完整交付基础

按“是否所有专项验收项都已严格闭环”看：
- 还差少量专项数据闭环

当前仍需注意的点：
- 正式白天 / 夜间专项验收仍建议补有标注的专项测试集
- 特种车牌、军牌、武警牌、临时牌专项验收仍建议补专项测试数据
- 如需与 `LPRNet` 形成完全同口径数值对照表，仍需补跑 `LPRNet` 基线实验

## 八、建议查看顺序

如果你后面自己查看项目，建议按这个顺序看：

1. [README.md](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\README.md)
2. [项目验收说明.md](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\analysis\项目验收说明.md)
3. [acceptance_summary.json](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\acceptance_results\acceptance_summary.json)
4. [compare_summary.json](E:\11xuexi\sx1\2gd\9\Sampling_of_CCPD_files\compare_ocr_results\compare_summary.json)
5. 检测权重和 OCR 权重
