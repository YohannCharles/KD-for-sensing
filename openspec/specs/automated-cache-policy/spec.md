# automated-cache-policy Specification

## Purpose
定义训练、评估和预热入口如何根据启用模态控制 RGB/ImageNet image-derived cache 与 LiDAR BEV cache，并持续拒绝已退役 image motion cache。
## Requirements
### Requirement: 统一 cache policy 配置
系统 MUST 提供统一的预处理 cache policy 配置，用于控制训练和评估入口对 LiDAR BEV cache 与 RGB/ImageNet image-derived cache 的读取、写入和重建行为。policy MUST 支持 `off`、`read_only`、`auto` 和 `rebuild`，并 MUST 允许按受支持模态覆盖全局 policy。系统 MUST 不提供已退役 image motion mask cache policy。

#### Scenario: 全局 auto policy
- **WHEN** 用户设置 `data.cache.policy: auto`
- **THEN** 系统 MUST 只为实际启用的 LiDAR 或 image modality 解析对应 cache
- **AND** cache miss 时 MUST 在线生成当前 BEV 或 RGB/ImageNet transform，并按 policy 决定是否写入
- **AND** image-derived cache MUST 不包含或恢复 image motion mask 语义

#### Scenario: read-only 与 off policy
- **WHEN** 用户设置 `read_only` 或 `off`
- **THEN** `read_only` MUST 允许命中已有匹配 cache、miss 时在线计算但不得写入
- **AND** `off` MUST 禁用对应 cache 的读取与写入并使用在线预处理

#### Scenario: 模态级 policy 覆盖
- **WHEN** 用户设置全局 `read_only` 且设置 `data.cache.image.policy: auto` 或 `data.cache.lidar.policy: auto`
- **THEN** 对应已启用模态 MUST 使用覆盖后的 policy
- **AND** 未启用模态 MUST 不访问其 cache

#### Scenario: image motion cache policy 被拒绝
- **WHEN** 用户配置 `image_motion_*`、`data.cache.image_motion` 或 `data.cache.image.motion_*` 字段
- **THEN** 配置解析 MUST 拒绝该配置
- **AND** 错误 MUST 指向 RGB/ImageNet image-derived cache 或关闭 image cache

### Requirement: 自动 cache policy 只作用于启用模态
系统 MUST 根据实际启用模态应用 cache policy。未启用 LiDAR 时不得访问 LiDAR BEV cache；未启用 image 时不得访问 image-derived cache。任何模态组合都不得读取已删除的 image motion cache。

#### Scenario: GPS-only 不访问 cache
- **WHEN** 用户运行 GPS-only 训练且 `data.cache.policy: auto`
- **THEN** dataset MUST 不检查、不创建、不读取、不写入 LiDAR 或 image-derived cache
- **AND** 缺失历史 image motion cache MUST 不阻止该任务运行

#### Scenario: image-only 使用可选 image-derived cache
- **WHEN** 用户运行 image-only 训练并启用 image cache policy
- **THEN** dataset MUST 使用与直接路径等价的 RGB/ImageNet tensor
- **AND** 系统 MUST 只访问版本和 fingerprint 匹配的 image-derived cache

#### Scenario: radar+mmWave fusion 不访问无关 cache
- **WHEN** 用户运行 `modalities: ["radar", "mmwave"]`
- **THEN** 自动 cache policy MUST 不访问 LiDAR 或 image-derived cache
- **AND** 缺失 image/LiDAR 原始文件或历史 cache MUST 不阻止该任务运行

### Requirement: cache policy 生效信息可追踪
训练、评估和预热运行 MUST 在最终配置或运行报告中记录实际生效的 cache policy、启用模态、受支持 cache 目录和相关 cache 的读写状态。系统 MUST 不记录 `image_motion_*` 字段，也 MUST 不要求 standalone training-I/O profile 产物。

#### Scenario: 训练记录 cache policy
- **WHEN** 一次训练运行构建 dataset
- **THEN** 最终配置或 metadata MUST 记录全局与受支持模态级 policy
- **AND** 启用 LiDAR 或 image 时 MUST 分别记录 BEV 或 image-derived cache 的目录、读写开关和 transform provenance
- **AND** 未启用模态 MUST 不产生对应 cache 访问记录

#### Scenario: 评估记录 cache policy
- **WHEN** 用户运行评估入口
- **THEN** 报告或最终配置 MUST 记录实际使用的受支持 cache policy
- **AND** MUST 不包含 `image_motion_cache_dir`、`image_motion_use_cache` 或 `image_motion_write_cache`

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
训练、评估和预热入口 MUST 在构建 dataset 前解析 cache policy，并将解析后的 cache 读写开关传递给 dataset。解析过程 MUST 使用实际启用模态，不得要求用户为每个组合手工设置低层开关；系统 MUST 不解析或传递 `image_motion_*` 字段。

#### Scenario: 单模态 image 训练解析 image-derived cache
- **WHEN** 用户运行 image-only 训练配置
- **THEN** 入口 MUST 使用 RGB/ImageNet profile，并按 `data.cache.image.policy` 解析 image-derived cache
- **AND** 入口 MUST 不生成或接受 `image_motion_*` 低层开关

#### Scenario: 任意 fusion 组合自动解析
- **WHEN** 用户运行任意 fusion 配置并声明 `modalities`
- **THEN** 入口 MUST 只为组合中受支持的 cache 模态解析行为
- **AND** 不包含 image 或 LiDAR 的组合 MUST 不需要相关 cache 参数即可启动

### Requirement: Cache overwrite 只删除可证明归属的 cache
任何 cache overwrite 或 rebuild 流程 MUST 只递归删除受控 cache root 下、具有匹配 owner marker 和预期结构的 cache 目录。系统 MUST NOT 对任意配置路径直接执行递归删除。

#### Scenario: 创建 cache ownership marker
- **WHEN** workflow 新建可覆盖 cache
- **THEN** cache root MUST 写入 machine-readable marker
- **AND** marker MUST 记录 schema version、owner、resolved cache path 和创建时间

#### Scenario: 无 marker 的历史目录
- **WHEN** overwrite 目标存在但没有有效 owner marker
- **THEN** runtime MUST 拒绝删除该目录
- **AND** MUST 提示用户选择新 cache 路径或显式迁移，而不是自动认领

#### Scenario: 项目或数据根误配置
- **WHEN** cache path 等于项目根、dataset root、outputs root 本身或允许 cache root 之外路径
- **THEN** overwrite MUST 在递归删除前失败
- **AND** 目标内容 MUST 保持不变

#### Scenario: Marker 与路径不匹配
- **WHEN** marker owner、resolved path、schema 或预期 cache 结构与当前请求不一致
- **THEN** overwrite MUST 拒绝删除
- **AND** 错误 MUST 说明不匹配字段

#### Scenario: 伪造 marker 或不完整 LMDB 结构
- **WHEN** overwrite 目标具有匹配 marker，但缺少 `data.mdb` 或 `lock.mdb`，或任一文件不是普通非符号链接文件
- **THEN** overwrite MUST 在递归删除前失败
- **AND** 目标目录及其现有内容 MUST 保持不变
