# OCR、LPRNet 与 PaddleOCR 对比分析

## 1. 文档目的

本文档用于支撑项目验收中的两项内容：

1. OCR 车牌识别算法分析
2. OCR 与之前 LPRNet 车牌识别方案的训练、测试指标对比分析

在项目推进过程中，又补充了一套 `PaddleOCR` 同数据集实验。因此当前对比已经从“两模型对比”扩展为“三模型对比”：

1. `LPRNet`
2. 自研 `OCR(CRNN + CTC)`
3. `PaddleOCR(CRNN 路线配置)`

## 2. 对比前提

为了保证对比尽量公平，这三条路线尽可能统一以下条件：

1. 使用同一套识别数据集：`ocr/`
2. 使用同一套训练 / 验证 / 测试划分
3. 输入都是裁剪后的车牌图像
4. 标签都是整牌字符序列
5. 主要关注以下指标：
   - 验证集整牌准确率
   - 验证集字符准确率或近似字符编辑指标
   - 测试集整牌准确率
   - 测试集字符准确率或近似字符编辑指标

需要说明的是：

1. `LPRNet` 与自研 `OCR(CRNN + CTC)` 使用的是完全统一的本地评测脚本
2. `PaddleOCR` 使用的是 PaddleOCR 框架自带评测口径
3. 因此 `PaddleOCR` 的“字符级指标”与本项目前两个模型的字符准确率不是完全同定义，但仍具有较强参考价值

## 3. 三种方案的定位

### 3.1 LPRNet

`LPRNet` 是经典车牌识别专用网络，特点是结构轻量、推理速度快、针对标准车牌场景有较强针对性。

主要思路：

1. 用轻量卷积网络提取车牌图像特征
2. 在宽度方向上聚合时序信息
3. 输出字符分类序列
4. 配合 `CTC Loss` 训练

### 3.2 自研 OCR(CRNN + CTC)

本项目当前主识别路线是 `CRNN + CTC`。

主要思路：

1. 用 `CNN` 提取车牌图像特征
2. 用序列建模模块学习字符顺序关系
3. 用 `CTC` 完成整牌序列训练与解码

这条路线的重点不是单字符分类，而是把整张车牌看成一个完整序列识别问题。

### 3.3 PaddleOCR

`PaddleOCR` 是成熟 OCR 框架，本次实验中采用的是识别任务配置，整体也是 `CRNN + CTC` 思路，但训练框架、数据读取、评测与模型实现来自 PaddleOCR 工程体系。

它的意义主要在于：

1. 作为现成工业 OCR 框架基线
2. 作为自研 OCR 方案之外的第三方对照
3. 用来证明自研 OCR 方案和成熟框架相比处于什么水平

## 4. 结构层面对比

### 4.1 LPRNet 的优点

1. 结构轻量，参数量较小
2. 推理速度快，部署成本低
3. 对标准车牌、固定分布数据集较友好

### 4.2 LPRNet 的不足

1. 对数据分布变化更敏感
2. 对复杂场景和复杂字符组合的适应性偏弱
3. 后续扩展到更多省份、更多场景、更多牌种时，泛化压力更大

### 4.3 自研 OCR(CRNN + CTC) 的优点

1. 更适合序列建模
2. 不依赖字符切分
3. 对汉字、字母、数字混合序列更自然
4. 更适合作为后续多省份、多场景扩展基础

### 4.4 PaddleOCR 的优点

1. 框架成熟，工程化程度高
2. 配套训练、评测、导出流程完整
3. 方便后续继续尝试更强识别模型

### 4.5 PaddleOCR 的不足

1. 本次直接迁移到车牌数据集后，效果没有超过自研 OCR
2. 默认框架口径和本项目现有评测口径不完全一致
3. 如果要继续优化，还需要进一步做配置微调与专项适配

## 5. 为什么本项目主路线仍选择 YOLOv11 + 自研 OCR

本项目需求明确要求识别部分采用 `OCR` 路线升级，因此关键不是“是否继续坚持 LPRNet”，而是“选择哪一种 OCR 路线最适合当前项目目标”。

从本次实验结果看：

1. `LPRNet` 已经被实测证明弱于当前 OCR 方案
2. `PaddleOCR` 作为成熟框架虽然可用，但在本次同数据集实验中也没有超过当前自研 OCR
3. 当前自研 `OCR(CRNN + CTC)` 在本项目数据上表现最好、链路最完整、与现有检测流程结合最稳定

因此当前项目主路线继续采用：

- `YOLOv11 + OCR(CRNN + CTC)`

## 6. 数据与结果文件来源

### 6.1 自研 OCR / LPRNet 统一识别数据

- 训练集标签：`ocr/train_label.txt`
- 验证集标签：`ocr/val_label.txt`
- 测试集标签：`ocr/test_label.txt`
- 字符集文件：`ocr/charset.txt`

### 6.2 自研 OCR / LPRNet 对比结果文件

- [runs_ocr/metrics.json](../runs_ocr/metrics.json)
- [runs_lprnet/metrics.json](../runs_lprnet/metrics.json)
- [recognizer_compare_summary.json](./recognizer_compare_summary.json)

### 6.3 PaddleOCR 结果文件

- PaddleOCR 训练数据目录：`paddleocr_rec/`
- PaddleOCR 训练输出目录：`paddleocr_output/plate_rec/`
- [train.log](../paddleocr_output/plate_rec/train.log)
- [config.yml](../paddleocr_output/plate_rec/config.yml)
- [best_accuracy.pdparams](../paddleocr_output/plate_rec/best_accuracy.pdparams)

## 7. 实测结果

### 7.1 自研 OCR(CRNN + CTC) 验证集结果

- `best_epoch = 64`
- `best_exact_accuracy = 0.9907`
- `best_char_accuracy = 0.9984`

### 7.2 LPRNet 验证集结果

- `best_epoch = 78`
- `best_exact_accuracy = 0.9750`
- `best_char_accuracy = 0.9867`

### 7.3 PaddleOCR 验证集结果

来源：`train.log` 中的最佳验证指标

- `best_epoch = 73`
- `best_metric_acc = 0.9557`
- `best_norm_edit_dis = 0.9917`

说明：

1. 这里的 `acc` 是 PaddleOCR 自带验证口径
2. `norm_edit_dis` 是归一化编辑距离指标，数值越高越好

### 7.4 自研 OCR / LPRNet 测试集统一评测结果

来源：`analysis/recognizer_compare_summary.json`

测试集样本数：

- `3000`

自研 OCR(CRNN + CTC)：

- `exact_accuracy = 0.9923`
- `char_accuracy = 0.9987`

LPRNet：

- `exact_accuracy = 0.9813`
- `char_accuracy = 0.9890`

### 7.5 PaddleOCR 测试集评测结果

来源：你补跑的 PaddleOCR 测试集 `eval.py` 输出

- `test_acc = 0.9480`
- `test_norm_edit_dis = 0.9895`

说明：

1. `PaddleOCR` 的测试集结果已经单独补跑
2. 其测试口径与前两个模型不是完全相同的本地自定义字符准确率口径
3. 但整牌识别趋势已经足够用于三模型方案比较

## 8. 三模型对比表

| 对比项 | LPRNet | 自研 OCR(CRNN + CTC) | PaddleOCR |
|---|---:|---:|---:|
| 训练数据 | `ocr/` | `ocr/` | `ocr/` 转 `paddleocr_rec/` |
| 验证集最优整牌准确率 | 0.9750 | 0.9907 | 0.9557 |
| 验证集字符级指标 | 0.9867 | 0.9984 | 0.9917(norm_edit_dis) |
| 测试集整牌准确率 | 0.9813 | 0.9923 | 0.9480 |
| 测试集字符级指标 | 0.9890 | 0.9987 | 0.9895(norm_edit_dis) |
| 是否作为当前项目主识别方案推荐 | 否 | 是 | 否 |

## 9. 对比结论

从当前已经补齐的三模型结果来看，可以得到以下结论：

1. 在本项目当前这套车牌识别数据上，自研 `OCR(CRNN + CTC)` 效果最好
2. `LPRNet` 作为传统专用网络，性能明显低于当前 OCR 主路线
3. `PaddleOCR` 作为成熟框架可以正常完成训练和评测，但在本次实验中也没有超过自研 OCR
4. 因此，从项目实测结果出发，当前主识别方案继续采用自研 `OCR(CRNN + CTC)` 是合理的

补充说明：

- `PaddleOCR` 已经完成同数据集对照实验，结果有效，可作为答辩中的第三方框架对照依据
- 但从当前实测结论看，它不作为本项目最终推荐落地识别模型

## 10. 对比的意义

补齐三模型对比之后，项目在识别方案选择上已经形成更完整闭环：

1. 不是只对比了历史 `LPRNet`
2. 不是只证明“换成 OCR 更好”
3. 而是进一步补了现成成熟 OCR 框架 `PaddleOCR`
4. 最终证明当前项目所采用的识别路线在本项目数据上是更优方案

