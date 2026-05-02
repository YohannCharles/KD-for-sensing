## ADDED Requirements

### Requirement: 按序列分层采样
诊断入口 MUST 支持 `diagnostics.visualization.per_seq_sample_count` 配置，用于从每个可用 `seq_index` 中可复现地抽取固定数量样本。启用该配置时，系统 MUST 在输出摘要中记录每个 `seq_index` 的候选数量、请求数量、实际选中数量和选中 dataset index。

#### Scenario: 覆盖每个 test seq_index
- **WHEN** 用户配置 `splits: ["test"]` 且设置 `per_seq_sample_count: 1`
- **THEN** 系统 MUST 从过滤后的每个可用 `seq_index` 中最多选择 1 个样本
- **AND** 输出摘要 MUST 按 `seq_index` 记录实际选中样本

#### Scenario: 分层采样候选不足
- **WHEN** 某个 `seq_index` 的候选数量少于 `per_seq_sample_count`
- **THEN** 系统 MUST 输出该 `seq_index` 下所有可用候选样本
- **AND** 系统 MUST 在 sampling 摘要中记录该 `seq_index` 的请求数量和实际数量

#### Scenario: 分层采样保持可复现
- **WHEN** 用户使用相同配置、相同 `per_seq_sample_count` 和相同 seed 两次运行诊断入口
- **THEN** 系统 MUST 为每个 `seq_index` 选择相同的 dataset index 集合
- **AND** 样本输出顺序 MUST 保持一致

### Requirement: 聚合 split 统计产物
诊断入口 MUST 写出 `split_stats.json`，用于按 scene、split 和 `seq_index` 汇总候选集合及处理后模态张量的诊断统计。该文件 MUST 至少包含 image mask density、radar RA/DA std、LiDAR nonzero fraction、future label 分布、majority baseline 和 train/test label 分布距离中当前启用模态和可用 split 对应的字段。

#### Scenario: 生成 split_stats.json
- **WHEN** 诊断入口成功完成一次运行
- **THEN** 系统 MUST 在输出目录写出 `split_stats.json`
- **AND** `summary.json` MUST 记录 `split_stats.json` 路径并将其加入输出文件列表

#### Scenario: 按 seq_index 汇总模态统计
- **WHEN** 诊断数据中存在多个 `seq_index`
- **THEN** `split_stats.json` MUST 为每个 split 下的每个 `seq_index` 记录样本数量和启用模态的聚合统计
- **AND** image、radar 和 LiDAR 字段 MUST 分别使用 Dataset 返回的 motion mask、RA/DA 张量和 BEV 张量计算

#### Scenario: 记录多数类基线和标签距离
- **WHEN** split CSV 中存在 future beam label
- **THEN** `split_stats.json` MUST 记录每个 split 的 future label top-k 分布和 majority baseline
- **AND** 当同一 scene 同时包含 train 与 test split 时，系统 MUST 记录 train/test future label 分布距离

### Requirement: 跨场景同口径比较
诊断入口 MUST 支持 `diagnostics.visualization.compare_scenes` 配置，用于在同一命令中按相同诊断参数处理多个 DeepSense6G scene。多 scene 运行 MUST 为每个 scene 使用独立输出目录，并在总摘要中记录各 scene 的 `summary.json`、`split_stats.json` 和实际输出目录。

#### Scenario: 同时比较 Scene 9 与 Scene 32
- **WHEN** 用户配置 `compare_scenes: [9, 32]`
- **THEN** 系统 MUST 分别覆盖 `data.dataset.scene` 为 9 和 32 并运行诊断
- **AND** 系统 MUST 生成一个总摘要，列出 Scene 9 与 Scene 32 的诊断产物路径

#### Scenario: 多场景输出目录隔离
- **WHEN** 多 scene 诊断使用同一个基础 `output_dir`
- **THEN** 系统 MUST 为每个 scene 写入互不冲突的 scene 子目录
- **AND** 不同 scene 的 PNG、样本列表、`summary.json` 和 `split_stats.json` MUST 不互相覆盖

### Requirement: 可读性改进的样本总览图
诊断入口生成的单样本 PNG 总览图 MUST 使用可读布局展示启用模态，并减少标题裁切、热力图过小和大量留白。图像 MUST 在启用对应模态时提供 raw image reference 与 processed motion mask 对照、radar RA/DA 共享色标展示，以及 LiDAR BEV/通道非零统计摘要。

#### Scenario: raw 与 processed image 同屏
- **WHEN** 样本启用 image 模态且用户开启 `include_raw_image_preview`
- **THEN** 单样本 PNG MUST 同时展示 raw image reference 和 Dataset 返回的 processed image motion mask
- **AND** 样本记录 MUST 标明 raw image 仅作为 reference

#### Scenario: radar 使用共享色标
- **WHEN** 样本启用 radar 模态
- **THEN** 单样本 PNG MUST 展示 RA 和 DA 的处理后 heatmap
- **AND** 同一 radar 子序列内的 heatmap MUST 使用一致色标，便于比较时间帧强度变化

#### Scenario: LiDAR 展示密度信息
- **WHEN** 样本启用 LiDAR 模态
- **THEN** 单样本 PNG MUST 展示 Dataset 返回的 LiDAR BEV 表示
- **AND** 图中或样本记录 MUST 包含总体非零率与通道级非零率摘要

### Requirement: 元数据产物不覆盖历史运行
诊断入口 MUST 默认保留已有机器可读元数据产物。当输出目录中已有本轮将写入的基础元数据文件时，系统 MUST 为本轮 `summary.json`、`samples.jsonl`、`samples.csv`、`split_stats.json` 和 `final_config.yaml` 选择同一个非冲突递增后缀，并在返回值、`summary.json` 和 `output_files` 中记录实际路径。用户 MUST 能通过配置 `diagnostics.visualization.preserve_existing_outputs: false` 恢复覆盖式写出。

#### Scenario: 后续运行不覆盖已有元数据
- **WHEN** 输出目录已存在 `summary.json`、`samples.jsonl`、`samples.csv`、`split_stats.json` 或 `final_config.yaml`
- **THEN** 新一次诊断运行 MUST 不覆盖这些已有文件
- **AND** 系统 MUST 写出同一批次后缀的元数据文件，例如 `summary_001.json`、`samples_001.jsonl`、`samples_001.csv`、`split_stats_001.json` 和 `final_config_001.yaml`

#### Scenario: 干净输出目录保留基础文件名
- **WHEN** 输出目录不存在基础元数据文件
- **THEN** 诊断入口 MUST 继续写出 `summary.json`、`samples.jsonl`、`samples.csv`、`split_stats.json` 和 `final_config.yaml`
- **AND** 返回值 MUST 记录这些基础路径

#### Scenario: 显式允许覆盖
- **WHEN** 用户配置 `diagnostics.visualization.preserve_existing_outputs: false`
- **THEN** 诊断入口 MUST 使用基础元数据文件名写出
- **AND** 若基础元数据文件已存在，系统 MAY 覆盖这些文件
