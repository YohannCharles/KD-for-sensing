## MODIFIED Requirements

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

## ADDED Requirements

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
