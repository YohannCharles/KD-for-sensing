# automated-cache-policy Specification

## Purpose
定义训练、评估和 profile 入口如何根据启用模态自动读取、写入、重建和记录 image motion mask cache 与 LiDAR BEV cache。
## Requirements
### Requirement: 统一 cache policy 配置
系统 MUST 提供统一的预处理 cache policy 配置，用于控制训练和评估入口对仍受支持的预处理 cache 的读取、写入和重建行为。policy MUST 至少支持 `off`、`read_only`、`auto` 和 `rebuild`，并 MUST 允许按受支持模态覆盖全局 policy。系统 MUST 不再提供 image motion mask cache policy。

#### Scenario: 全局 auto policy
- **WHEN** 用户设置 `data.cache.policy: auto`
- **THEN** 系统 MUST 对启用 LiDAR 的 dataset 自动启用 LiDAR BEV cache 读取
- **AND** 系统 MUST 在 LiDAR BEV cache miss 时按需生成并写入对应 cache
- **AND** 启用 image modality MUST 不触发 image motion cache 读取、写入或目录创建

#### Scenario: read-only policy
- **WHEN** 用户设置 `data.cache.policy: read_only`
- **THEN** 系统 MUST 对启用 LiDAR 的 dataset 自动尝试读取已有 LiDAR BEV cache
- **AND** LiDAR BEV cache miss 时系统 MUST 在线计算当前样本所需预处理结果但不得写入新 cache
- **AND** 系统 MUST 不解析 image motion cache 的 read-only 行为

#### Scenario: off policy
- **WHEN** 用户设置 `data.cache.policy: off`
- **THEN** 系统 MUST 禁用 LiDAR BEV cache 的读取与写入
- **AND** 系统 MUST 使用在线 LiDAR 预处理路径完成训练或评估
- **AND** 系统 MUST 不要求 image motion cache 字段存在

#### Scenario: 模态级 policy 覆盖
- **WHEN** 用户设置全局 `data.cache.policy: read_only` 且设置 `data.cache.lidar.policy: auto`
- **THEN** LiDAR cache MUST 使用 `auto` 行为
- **AND** image modality MUST 没有可覆盖的 image motion cache policy

#### Scenario: image motion cache policy 被拒绝
- **WHEN** 用户配置 `data.cache.image.policy`、`image_motion_use_cache`、`image_motion_write_cache` 或其它 `image_motion_*` cache 字段
- **THEN** 配置解析 MUST 拒绝该配置
- **AND** 错误信息 MUST 说明 image motion cache 已删除且不提供兼容回退

### Requirement: 自动 cache policy 只作用于启用模态
系统 MUST 根据实际启用模态应用 cache policy。未启用 LiDAR 时不得访问 LiDAR BEV cache。image modality 启用时也不得访问 image motion cache，因为该 cache 类型已删除。

#### Scenario: GPS-only 不访问 cache
- **WHEN** 用户运行 GPS-only 训练且 `data.cache.policy: auto`
- **THEN** dataset MUST 不检查、不创建、不读取、不写入 LiDAR BEV cache
- **AND** dataset MUST 不检查、不创建、不读取、不写入 image motion cache

#### Scenario: image-only 不访问 image motion cache
- **WHEN** 用户运行 image-only 训练且 `data.cache.policy: auto`
- **THEN** dataset MUST 使用 RGB/ImageNet image 输入
- **AND** dataset MUST 不检查、不创建、不读取、不写入 image motion cache
- **AND** 缺失旧 image motion cache MUST 不阻止该任务运行

#### Scenario: radar+mmWave fusion 不访问 LiDAR cache
- **WHEN** 用户运行 fusion 配置且 `modalities: ["radar", "mmwave"]`
- **THEN** 自动 cache policy MUST 不访问 LiDAR BEV cache
- **AND** 缺失 image/LiDAR 原始文件或旧 image cache MUST 不阻止该任务运行

### Requirement: cache policy 生效信息可追踪
训练、评估和预热运行 MUST 在最终配置或运行报告中记录实际生效的 cache policy、启用模态、受支持 cache 目录和每个相关 cache 的读写状态。系统 MUST 不再记录 `image_motion_*` cache 字段，也 MUST 不要求 standalone training-I/O profile 产物。

#### Scenario: 训练记录 cache policy
- **WHEN** 一次训练运行构建 train/test dataset
- **THEN** `final_config.yaml` 或训练日志 MUST 记录全局 cache policy 和受支持模态级 policy
- **AND** 对启用 LiDAR 的配置 MUST 记录 LiDAR BEV cache 目录、实际读取开关和实际写入开关
- **AND** 对启用 image 的配置 MUST 不记录 image motion cache 目录或读写开关

#### Scenario: 评估记录 cache policy
- **WHEN** 用户运行评估入口
- **THEN** 测试报告或最终配置 MUST 记录评估时实际使用的 cache policy 和受支持 cache 目录
- **AND** 测试报告或最终配置 MUST 不包含 `image_motion_cache_dir`、`image_motion_use_cache` 或 `image_motion_write_cache`

### Requirement: MMW RGB/ImageNet image 派生缓存
系统 MUST 支持受控的 RGB/ImageNet image 派生缓存，用于 MMW/DeepSense6G image modality 的 processed image 输入。该 cache MUST 只表示当前 RGB/ImageNet image profile 的模型输入，不得复用或恢复已删除的 image motion cache。

#### Scenario: auto policy 生成 image-derived cache
- **WHEN** 用户启用 image modality 且设置 `data.cache.image.policy: auto`
- **THEN** dataset MUST 在 cache miss 时在线读取原始 image 并生成 RGB/ImageNet processed cache
- **AND** 后续访问同一 image、image size 和 transform version MUST 能复用该 cache
- **AND** 返回 image tensor 的 shape、dtype 和数值语义 MUST 与未启用 cache 时一致

#### Scenario: read-only policy 不写入 image cache
- **WHEN** 用户设置 `data.cache.image.policy: read_only`
- **THEN** dataset MUST 读取已有且 fingerprint 匹配的 image-derived cache
- **AND** cache miss 或 fingerprint 不匹配时 MUST 在线计算当前样本
- **AND** 系统 MUST 不写入新的 image-derived cache 文件

#### Scenario: image motion cache 仍被拒绝
- **WHEN** 用户配置 `image_motion_*` 或旧 image motion cache 字段
- **THEN** 配置解析 MUST 拒绝该配置
- **AND** 错误信息 MUST 说明应使用 RGB/ImageNet image-derived cache 或关闭 image cache

### Requirement: image-derived cache 可追踪
训练、评估和预热入口 MUST 记录 image-derived cache 的生效策略、cache 目录、transform version、coverage、命中/缺失统计和生成行为。未启用 image modality 时不得访问 image-derived cache；standalone training-I/O profile 不再是 required consumer。

#### Scenario: 运行产物记录 image cache 状态
- **WHEN** 一次训练或评估构建启用 image modality 的 dataset
- **THEN** 运行 metadata MUST 记录 image cache policy、cache dir、transform version、hit/miss 或 coverage 摘要
- **AND** metadata MUST 不包含旧 image motion cache 字段

#### Scenario: 未启用 image 不访问 image cache
- **WHEN** 用户运行 GPS+mmWave 或其它不包含 image 的配置
- **THEN** cache policy MUST 不检查、不创建、不读取、不写入 image-derived cache
- **AND** 缺失 image cache 不得阻止该任务运行

### Requirement: image-derived cache 预热
项目 MUST 提供可选的 image-derived cache 预热能力。预热 MUST 遵守 dataset split、enabled modalities、image profile、image size 和 cache policy，不得把生成的缓存纳入源码变更。

#### Scenario: 预热指定 split
- **WHEN** 用户运行 image-derived cache 预热入口并指定 MMW train split
- **THEN** 系统 MUST 为该 split 中启用 image modality 的样本生成 cache
- **AND** 预热报告 MUST 记录扫描样本数、生成数、跳过数、失败数、cache 总大小和输出目录

#### Scenario: 预热不改变样本契约
- **WHEN** 同一样本分别通过原始 image 路径和 image-derived cache 路径读取
- **THEN** 返回 tensor 的 shape、dtype、模态字段和 target 字段 MUST 保持一致
- **AND** focused tests MUST 覆盖该等价性

### Requirement: 实验入口自动解析 cache policy
训练、评估和预热入口 MUST 在构建 dataset 前解析 cache policy，并将解析后的实际 cache 读写开关传递给 dataset。解析过程 MUST 使用配置中的启用模态，不得要求用户为每个单模态或 fusion 组合手动设置低层 cache 读写字段。系统 MUST 不再解析或传递 `image_motion_*` 低层开关，也 MUST 不维持 standalone training-I/O profile 入口。

#### Scenario: 单模态 image 训练不解析 image motion cache
- **WHEN** 用户运行 image-only 训练配置
- **THEN** 训练入口 MUST 使用 RGB/ImageNet image 输入构建 dataset
- **AND** 训练入口 MUST 不生成 `image_motion_use_cache` 或 `image_motion_write_cache`
- **AND** 用户 MUST 不能通过命令行恢复这些已删除低层开关

#### Scenario: 任意 fusion 组合自动解析
- **WHEN** 用户运行任意 fusion 配置并声明 `modalities`
- **THEN** 训练入口 MUST 只为该组合包含的受支持 cache 模态解析 cache 行为
- **AND** 不包含 LiDAR 的组合 MUST 不需要相关 cache 参数才能启动
- **AND** 包含 image 的组合 MUST 不需要且不得接受 image motion cache 参数
