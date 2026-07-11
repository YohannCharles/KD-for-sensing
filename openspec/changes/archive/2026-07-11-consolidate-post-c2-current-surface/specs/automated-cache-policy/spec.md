## MODIFIED Requirements

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

## REMOVED Requirements

### Requirement: Benchmark perturbation cache
**Reason**: 该 cache 只服务已退役 JEPA GPS shortcut/Scenario C/D benchmark，当前源码没有 producer 或 consumer。
**Migration**: Current evaluation 使用在线 difficulty pipeline；历史 benchmark cache contract 从 archive 查询。

#### Scenario: Retired benchmark cache 不再解析
- **WHEN** config 或 manifest 请求旧 perturbation cache mode
- **THEN** current config/runner MUST 不恢复该 benchmark path
- **AND** source dataset、labels 和 runtime artifact boundaries MUST 保持不变

### Requirement: Cached benchmark comparability
**Reason**: 对应 perturbation cache 整体退出，专属 comparability schema 不再需要。
**Migration**: Current evaluation comparability 继续由 metric/difficulty provenance owner管理。

#### Scenario: Current comparability 不依赖旧 cache
- **WHEN** current evaluation 比较多个 run
- **THEN** comparability MUST 不要求旧 perturbation cache schema 或 hit/miss 字段
- **AND** split、label space、sample ids、metric profile 和 difficulty digest 仍由 current owner校验
