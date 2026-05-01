## Why

Scene 32 中 image/radar/LiDAR 相关组合在采用 `balanced_seq` 后仍出现训练精度上升、验证精度下降，说明问题不太可能只来自旧的顺序 split。需要一个复用真实数据加载和预处理路径的诊断可视化工具，直接检查各模态处理后的输入在 Scene 9、Scene 32、train/test 和不同 `seq_index` 间是否存在信息丢失或域偏移。

## What Changes

- 新增一个各模态处理后可视化诊断能力，支持从现有实验配置或诊断配置构建 DeepSense6G dataset。
- 支持按 scene、split、`seq_index`、label、样本数量和随机 seed 抽样，并输出可复现的诊断产物。
- 对同一窗口输出 image motion mask、radar RA/DA、LiDAR BEV、GPS relative-polar 轨迹、mmWave receive-power 特征、历史 beam 和 future beam label 的并排视图。
- 输出每个诊断批次的 JSON/CSV 摘要，记录样本索引、路径、label、`seq_index`、启用模态和关键统计量，例如 image mask density、radar 强度摘要、LiDAR 非零率、GPS 范围、mmWave 均值/方差。
- 提供默认配置和 CLI/script 入口，便于比较 Scene 9 与 Scene 32、train 与 test、以及失败模态组合与正常模态组合。
- 不改变训练、评估、模型结构、loss、split 协议或现有 cache 内容。

## Capabilities

### New Capabilities

- `modality-visual-diagnostics`: 定义如何生成 DeepSense6G 各模态处理后样本的可视化和统计诊断产物。

### Modified Capabilities

无。

## Impact

- 受影响代码：新增诊断 CLI/script、配置文件、可视化工具函数和相关测试。
- 复用代码：`DeepSense6GDataset`、模态 transforms、cache policy、split metadata 读取和路径解析。
- 受影响产物：新增诊断输出目录，包含 PNG 图像网格、统计 JSON/CSV 和运行配置快照。
- 依赖影响：优先使用项目已有 `matplotlib`、`numpy`、`pandas`、`torch` 和 `PIL/skimage`；不引入新的训练时依赖。
