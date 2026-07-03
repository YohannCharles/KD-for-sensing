# dataset-directory-layout Specification

## Purpose
定义本地数据集家族目录布局、可生成数据边界和路径解析约束，避免源码、原始数据和运行产物之间的责任混淆。
## Requirements
### Requirement: 数据集家族目录规范
项目 MUST 在 `dataset/` 下按数据集家族组织本地数据。每个数据集家族 MUST 拥有独立的一级目录，场景、天气或条件等细分 MUST 放在对应家族目录内部。

#### Scenario: DeepSense6G 使用家族目录
- **WHEN** 用户使用未显式覆盖 `data_root` 的 DeepSense6G 配置
- **THEN** 默认数据根目录 MUST 位于 `dataset/DeepSense6G/` 下
- **AND** Scenario 31、Scenario 32 和 Scenario 9 MUST 分别使用 `scenario31`、`scenario32` 和 `scenario9` 子目录

#### Scenario: 数据集家族平级
- **WHEN** 项目同时存在 DeepSense6G 和 MMW 本地数据
- **THEN** DeepSense6G 数据 MUST 位于 `dataset/DeepSense6G/`
- **AND** MMW 数据 MUST 位于 `dataset/MMW/`

### Requirement: Dataset layout descriptor
项目 MUST 提供集中式 dataset layout descriptor 或等价机制，统一描述数据集家族、规范根目录、旧兼容根目录、场景/条件别名和必要子目录。训练、评估和预处理代码 MUST 复用该机制，不得在多个入口重复硬编码同一默认目录规则。

#### Scenario: 解析 DeepSense6G 规范根目录
- **WHEN** 代码请求 DeepSense6G Scenario 31 的默认根目录
- **THEN** layout descriptor MUST 返回 `dataset/DeepSense6G/scenario31`
- **AND** 返回结果 MUST 可被现有 `resolve_path` 按项目根目录解析

#### Scenario: 记录旧兼容根目录
- **WHEN** 代码请求 DeepSense6G Scenario 31 的 legacy 根目录
- **THEN** layout descriptor MUST 能提供 `dataset/scenario31`
- **AND** legacy 根目录 MUST 只用于显式覆盖、迁移文档或兼容性校验，不得作为未显式配置时的默认值

### Requirement: MMW 天气目录规范
未来 MMW 数据 MUST 按天气条件组织为 `dataset/MMW/<condition>/Sensor_Data` 和 `dataset/MMW/<condition>/Channel_Data`。首批条件名 MUST 包含 `sunny`、`rainy` 和 `foggy`。

#### Scenario: sunny MMW 目录
- **WHEN** 用户准备 sunny 条件的 MMW 数据
- **THEN** 传感器数据 MUST 放入 `dataset/MMW/sunny/Sensor_Data`
- **AND** 信道数据 MUST 放入 `dataset/MMW/sunny/Channel_Data`

#### Scenario: rainy 和 foggy MMW 目录
- **WHEN** 用户准备 rainy 或 foggy 条件的 MMW 数据
- **THEN** 对应条件目录 MUST 分别为 `dataset/MMW/rainy` 或 `dataset/MMW/foggy`
- **AND** 每个条件目录 MUST 包含 `Sensor_Data` 与 `Channel_Data` 两个语义子目录

### Requirement: 数据文件不自动迁移
代码变更 MUST 不自动移动、复制或删除 `dataset/` 下的真实数据文件。目录迁移 MUST 由用户显式执行，或通过显式配置 `data_root` 继续使用旧目录。新生成的可再生 cache SHOULD 默认写入 `outputs/cache/`，而不是数据集原始目录；显式配置到 `dataset/` 下的历史 cache 路径 MAY 继续使用。

#### Scenario: 默认配置不移动旧数据
- **WHEN** 用户运行默认 DeepSense6G 配置且本地存在旧 `dataset/scenario31`
- **THEN** 系统 MUST 不自动移动该目录
- **AND** 系统 MUST 不删除、覆盖或复制旧目录中的文件

#### Scenario: 显式旧路径继续可用
- **WHEN** 用户设置 `data.dataset.data_root: dataset/scenario31`
- **THEN** 系统 MUST 使用该显式路径构建 dataset
- **AND** 该行为 MUST 不依赖 `dataset/DeepSense6G/scenario31` 是否存在

#### Scenario: 新 cache 默认不写入 dataset
- **WHEN** DeepSense6G 或 MMW cache 路径未显式配置
- **THEN** 系统 MUST 将可再生 cache 默认写入 `outputs/cache/` 下的数据集 family 子目录
- **AND** 系统 MUST 不在 `dataset/` 下自动新建默认 cache 目录

### Requirement: Runtime cache layout descriptor
项目 MUST 提供集中式 runtime cache layout descriptor 或等价机制，用于描述可再生成 cache 的默认根目录和语义子目录。默认 runtime cache 根 MUST 为 `outputs/cache`；数据集相关 cache MUST 按 family 和 scene/condition 分层；非数据集专属 cache MUST 使用明确的 cache kind 子目录。

#### Scenario: DeepSense6G cache 默认路径
- **WHEN** 代码请求 DeepSense6G Scenario 31 的 image-derived 或 LiDAR BEV cache 默认路径
- **THEN** layout descriptor MUST 返回 `outputs/cache/DeepSense6G/scenario31/image_derived` 或 `outputs/cache/DeepSense6G/scenario31/lidar_bev`

#### Scenario: MMW cache 默认路径
- **WHEN** 代码请求 MMW sunny 条件的 image-derived 或 LiDAR BEV cache 默认路径
- **THEN** layout descriptor MUST 返回 `outputs/cache/MMW/sunny/image_derived` 或 `outputs/cache/MMW/sunny/lidar_bev`

#### Scenario: Root physical label cache 收敛
- **WHEN** MMW physical label cache 未显式配置 cache_dir
- **THEN** 系统 MUST 默认使用 `outputs/cache/physical_labels`
- **AND** 系统 MUST 不默认创建根目录 `cache/physical_labels`

### Requirement: Dataset layout audit
Dataset layout descriptor MUST 支持 dataset audit 读取 canonical root、legacy root、scene/condition alias 和 required subdirectories。Audit MUST 使用该 descriptor，而不是在多个脚本中重复硬编码路径。

#### Scenario: audit 解析 canonical layout
- **WHEN** audit 检查 DeepSense6G Scenario 31 或 MMW sunny condition
- **THEN** 它 MUST 通过 dataset layout descriptor 解析 canonical data root 和 expected subdirectories
- **AND** 报告 MUST 标记路径存在、缺失或显式 override

#### Scenario: legacy layout 标记为兼容输入
- **WHEN** audit 发现 legacy `dataset/scenario31` 或其它旧兼容根
- **THEN** 报告 MUST 标记为 legacy-compatible input
- **AND** audit MUST 不自动迁移、复制、删除或重命名该目录

#### Scenario: cache 与数据目录区分
- **WHEN** audit 发现 image/LiDAR/radar/cache-like artifacts
- **THEN** 报告 MUST 区分 raw data、prepared data、runtime cache 和 output artifact
- **AND** 可再生成 cache SHOULD 建议位于 `outputs/cache/`，但 audit MUST 不移动文件

