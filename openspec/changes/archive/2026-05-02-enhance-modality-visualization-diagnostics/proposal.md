## Why

现有模态可视化入口已经能输出少量样本图和样本级统计，但不足以解释 Scene 32 中 image/radar/LiDAR 组合停留在多数类基线的问题。当前输出缺少按 scene、split、`seq_index` 聚合的诊断统计、分层采样和更清晰的 raw/processed 对照，因此难以判断问题来自模态信号弱、train/test 域差异，还是预处理信息损失。

## What Changes

- 扩展可视化诊断配置，支持 `per_seq_sample_count` 分层采样，确保指定 split 中每个 `seq_index` 可被覆盖。
- 新增 `split_stats.json`，按 scene、split、`seq_index` 汇总 image mask density、radar RA/DA std、LiDAR nonzero fraction、label 分布、majority baseline 和 train/test label 分布距离。
- 支持 `compare_scenes`，一次运行中输出多个 DeepSense6G scene 的同口径诊断统计和样本图。
- 改进单样本 PNG 布局，使用更大的热力图、受约束布局、短标题和按模态拆分的行结构，减少留白、裁切和信息拥挤。
- 增强 image/radar/LiDAR 面板：raw image 与 motion mask 同屏、radar RA/DA 使用共享色标、LiDAR 输出通道统计与 BEV 非零信息。
- 诊断元数据产物默认保留历史运行结果；当输出目录已有 `summary.json`、`samples.jsonl`、`samples.csv`、`split_stats.json` 或 `final_config.yaml` 时，本次运行 MUST 写入同一批次的递增后缀文件，避免后续程序覆盖前一次诊断记录。
- 保留现有只读行为、Dataset 真实张量复用、样本列表、`summary.json` 和最终配置快照，不修改训练、评估、模型结构或 cache 写入策略。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `modality-visual-diagnostics`: 增加聚合 split/scene 统计、按序列分层采样、跨场景比较和改进后的样本图布局要求。

## Impact

- 受影响代码：`src/kd_sensing/diagnostics/modality_visualization.py`、`configs/diagnostics/modality_visualization.yaml`、相关 CLI 输出和测试。
- 受影响产物：诊断输出目录将新增 `split_stats.json`，`summary.json` 将记录跨场景诊断与聚合统计路径；重复运行时元数据文件会使用一致的 `_NNN` 后缀避免覆盖。
- 兼容性：现有配置字段保持兼容；新增字段均为可选，默认行为继续可生成少量 train/test 样本。
- 依赖影响：继续使用项目已有 `matplotlib`、`numpy`、`pandas`、`torch` 和 `PIL`，不引入新的运行时依赖。
