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
训练、评估和 profile 运行 MUST 在最终配置或运行报告中记录实际生效的 cache policy、启用模态、受支持 cache 目录和每个相关 cache 的读写状态。系统 MUST 不再记录 `image_motion_*` cache 字段。

#### Scenario: 训练记录 cache policy
- **WHEN** 一次训练运行构建 train/test dataset
- **THEN** `final_config.yaml` 或训练日志 MUST 记录全局 cache policy 和受支持模态级 policy
- **AND** 对启用 LiDAR 的配置 MUST 记录 LiDAR BEV cache 目录、实际读取开关和实际写入开关
- **AND** 对启用 image 的配置 MUST 不记录 image motion cache 目录或读写开关

#### Scenario: 评估记录 cache policy
- **WHEN** 用户运行评估入口
- **THEN** 测试报告或最终配置 MUST 记录评估时实际使用的 cache policy 和受支持 cache 目录
- **AND** 测试报告或最终配置 MUST 不包含 `image_motion_cache_dir`、`image_motion_use_cache` 或 `image_motion_write_cache`

