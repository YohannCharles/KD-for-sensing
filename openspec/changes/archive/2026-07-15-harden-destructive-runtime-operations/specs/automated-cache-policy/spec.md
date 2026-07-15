## ADDED Requirements

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

## MODIFIED Requirements

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
