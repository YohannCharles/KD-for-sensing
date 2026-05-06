# modality-visual-diagnostics Specification

## Purpose
Define the processed-modality visualization diagnostics entry point, including configuration parsing, sample selection, dataset reuse, output files, metadata stability, and scene comparison behavior.
## Requirements
### Requirement: 诊断入口与配置
系统 MUST 提供配置驱动的各模态处理后可视化诊断入口。该入口 MUST 能读取现有训练配置，并通过 `diagnostics.visualization` 或命令行覆盖控制 scene、split、启用模态、抽样条件、样本数量、随机 seed 和输出目录。

#### Scenario: 使用现有训练配置启动诊断
- **WHEN** 用户运行诊断入口并传入一个现有 image、radar、GPS、LiDAR、mmWave 或 fusion 训练配置
- **THEN** 系统 MUST 复用该配置中的 dataset、scene、train/test CSV、序列长度、预测长度、启用模态和 cache policy 构建诊断数据源
- **AND** 系统 MUST 将诊断产物写入配置或默认规则指定的输出目录

#### Scenario: 命令行覆盖诊断字段
- **WHEN** 用户通过命令行覆盖 `diagnostics.visualization.sample_count`、`diagnostics.visualization.seed` 或 `data.dataset.scene`
- **THEN** 系统 MUST 使用覆盖后的值执行抽样和输出
- **AND** 输出摘要 MUST 记录最终生效的诊断配置

#### Scenario: 未配置诊断字段
- **WHEN** 训练配置没有 `diagnostics.visualization` 字段
- **THEN** 诊断入口 MUST 使用安全默认值生成少量 train/test 样本
- **AND** 默认执行不得修改训练配置文件或训练输出目录中的既有文件

### Requirement: 复用真实处理后张量
诊断可视化 MUST 基于 Dataset 实际返回的处理后张量生成，确保可视化内容与训练、验证或评估输入一致。系统 MUST 不用单独的旁路预处理逻辑替代 Dataset 的 image motion mask、radar RA/DA、LiDAR BEV、GPS 和 mmWave 处理。

#### Scenario: image 可视化使用 motion mask 张量
- **WHEN** 诊断样本启用 image 模态
- **THEN** 输出图像 MUST 展示 Dataset 返回的 `image` motion mask 张量
- **AND** 输出摘要 MUST 记录该张量的 shape、dtype、min、max、mean、std 和 nonzero fraction

#### Scenario: radar 可视化使用 RA/DA 张量
- **WHEN** 诊断样本启用 radar 模态
- **THEN** 输出图像 MUST 展示 Dataset 返回的 `radar_ra` 和 `radar_da` 张量
- **AND** 输出摘要 MUST 分别记录 RA 和 DA 的 shape、dtype、min、max、mean、std 和 nonzero fraction

#### Scenario: LiDAR 可视化使用 BEV 张量
- **WHEN** 诊断样本启用 LiDAR 模态
- **THEN** 输出图像 MUST 展示 Dataset 返回的 `lidar` BEV 张量
- **AND** 输出摘要 MUST 记录 BEV 总体和每个通道的非零率或等价统计

#### Scenario: GPS 和 mmWave 可视化使用数值序列
- **WHEN** 诊断样本启用 GPS 或 mmWave 模态
- **THEN** 输出图像 MUST 将 Dataset 返回的 GPS 或 mmWave 张量显示为轨迹、曲线或 heatmap
- **AND** 输出摘要 MUST 记录对应张量的 shape、dtype、min、max、mean 和 std

### Requirement: 可复现样本选择
诊断入口 MUST 支持可复现地选择样本。用户 MUST 能按 split、`seq_index`、future beam label、样本数量和 seed 限定候选集合；输出摘要 MUST 记录最终选中的 dataset index、CSV 行信息、`seq_index` 和 label。

#### Scenario: 按 seq_index 过滤样本
- **WHEN** 用户指定一个或多个 `seq_index`
- **THEN** 诊断入口 MUST 只从这些 `seq_index` 对应的窗口中选择样本
- **AND** 输出摘要 MUST 记录每个样本所属的 `seq_index`

#### Scenario: 按 future beam label 过滤样本
- **WHEN** 用户指定一个或多个 future beam label
- **THEN** 诊断入口 MUST 只选择目标 label 匹配的样本
- **AND** 输出摘要 MUST 记录每个样本的历史 beam label 和 future beam label

#### Scenario: 相同 seed 可复现
- **WHEN** 用户用相同配置和相同 seed 两次运行诊断入口
- **THEN** 系统 MUST 选择相同的 dataset index 集合
- **AND** 输出摘要中的样本顺序 MUST 保持一致

#### Scenario: 候选样本不足
- **WHEN** 过滤条件下可用样本数少于请求的样本数量
- **THEN** 诊断入口 MUST 输出所有可用候选样本
- **AND** 系统 MUST 在摘要中记录请求数量和实际输出数量

### Requirement: 诊断产物结构
诊断入口 MUST 生成静态文件产物，至少包括单样本 PNG 总览图、样本级 CSV 或 JSON 列表、运行级 `summary.json` 和最终生效配置快照。产物路径 MUST 稳定且可由用户从摘要中定位。

#### Scenario: 生成单样本总览图
- **WHEN** 诊断入口成功处理一个样本
- **THEN** 系统 MUST 为该样本写出一个 PNG 总览图
- **AND** 图中 MUST 包含启用模态的处理后表示和该样本的 beam label 摘要

#### Scenario: 生成样本列表
- **WHEN** 诊断入口完成一次运行
- **THEN** 系统 MUST 写出机器可读的样本列表文件
- **AND** 样本列表 MUST 包含每个样本的 split、dataset index、`seq_index`、future beam label、输出 PNG 路径和启用模态统计摘要

#### Scenario: 生成运行摘要
- **WHEN** 诊断入口完成一次运行
- **THEN** 系统 MUST 写出 `summary.json`
- **AND** `summary.json` MUST 记录 scene、split、CSV 路径、样本数量、启用模态、抽样条件、seed、输出文件列表和 split metadata 摘要是否可用

### Requirement: 跨 split 与跨场景比较
诊断入口 MUST 支持在同一运行中处理多个 split，并 MUST 支持通过配置覆盖比较不同 DeepSense6G 场景。系统 MUST 在输出目录和摘要中清楚区分 scene 与 split，避免 Scene 9 与 Scene 32 产物混淆。

#### Scenario: 同时诊断 train 和 test
- **WHEN** 用户配置 `splits: ["train", "test"]`
- **THEN** 系统 MUST 分别构建 train 和 test 数据源并输出对应样本图
- **AND** 输出摘要 MUST 按 split 分组记录样本数量和统计摘要

#### Scenario: 诊断 Scene 9
- **WHEN** 用户将 `data.dataset.scene` 覆盖为 9
- **THEN** 诊断入口 MUST 使用 Scene 9 的数据根目录和 split CSV 构建数据源
- **AND** 输出摘要 MUST 记录 `scene_id: 9` 和 `scene_slug: scene9`

#### Scenario: 诊断 Scene 32
- **WHEN** 用户将 `data.dataset.scene` 覆盖为 32
- **THEN** 诊断入口 MUST 使用 Scene 32 的数据根目录和 split CSV 构建数据源
- **AND** 输出摘要 MUST 记录 `scene_id: 32` 和 `scene_slug: scene32`

### Requirement: 只读诊断行为
诊断入口 MUST 默认保持只读行为，不得修改训练 checkpoint、训练日志、评估报告或已存在的 split CSV。对 image motion cache 和 LiDAR BEV cache 的访问 MUST 遵循现有 cache policy；当 policy 为 `read_only` 或 `off` 时，诊断入口不得写入新的 cache 文件。

#### Scenario: 不修改训练产物
- **WHEN** 用户对已有训练配置运行诊断入口
- **THEN** 系统 MUST 不修改该训练运行目录中的 checkpoint、`train_log.json`、`metrics.json` 或 `final_config.yaml`
- **AND** 所有诊断产物 MUST 写入诊断输出目录

#### Scenario: read_only cache 不写入
- **WHEN** 用户设置 `data.cache.policy: read_only`
- **THEN** 诊断入口 MUST 允许读取已有 image motion mask cache 或 LiDAR BEV cache
- **AND** cache miss 时系统 MUST 在线计算当前样本所需处理结果但不得写入新 cache 文件

#### Scenario: off cache 不访问 cache 文件
- **WHEN** 用户设置 `data.cache.policy: off`
- **THEN** 诊断入口 MUST 禁用 image motion mask cache 和 LiDAR BEV cache 的读取与写入
- **AND** 系统 MUST 仍能通过在线处理生成诊断图

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
