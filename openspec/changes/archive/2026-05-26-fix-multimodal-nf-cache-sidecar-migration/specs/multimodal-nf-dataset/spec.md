## ADDED Requirements

### Requirement: Multimodal-NF 旧派生缓存 sidecar 迁移
Multimodal-NF image/LiDAR 派生缓存 MUST 支持将可验证的旧 sidecar 元数据升级为当前轻量校验 schema，而不重写对应 `.npy` 数据文件。系统 MUST 只有在旧 sidecar、`.npy` header、当前 source identity 和配置参数足以确认 cache 仍适用于当前 profile、split、`seq_len` 和 `num_pred` 时执行 metadata-only upgrade。

#### Scenario: 预处理升级旧 sidecar 不重写数据
- **WHEN** 用户运行 Multimodal-NF derived cache 预处理且 `rebuild=false`，并且某个 image 或 LiDAR `.npy` 文件存在、旧 sidecar 可迁移
- **THEN** 系统 MUST 补齐当前 cache schema 所需的 lightweight metadata 字段
- **AND** 系统 MUST 使用原子写入更新 sidecar JSON
- **AND** 系统 MUST NOT 重写对应 `.npy` 数据文件
- **AND** 预处理输出 MUST 记录该 source 的结果为 metadata upgraded 或等价机器可读状态

#### Scenario: auto 策略优先执行 metadata-only upgrade
- **WHEN** 训练或 dataset 构建使用 Multimodal-NF image/LiDAR cache `policy=auto`，并且所需 cache 的 `.npy` 文件存在但 sidecar 为可迁移旧 schema
- **THEN** 系统 MUST 优先执行 metadata-only sidecar upgrade
- **AND** 系统 MUST 在 upgrade 后重新执行 lightweight cache status 校验
- **AND** 系统 MUST 只有在 metadata-only upgrade 不安全或失败时才按现有策略重建 cache 或回退到原始 HDF5

#### Scenario: read_only 不自动写 sidecar
- **WHEN** 用户配置 Multimodal-NF image/LiDAR cache `policy=read_only`，并且所需 cache 的 `.npy` 文件存在但 sidecar 为可迁移旧 schema
- **THEN** dataset 构建 MUST 失败并保持 sidecar 不变
- **AND** 错误信息 MUST 明确说明 cache data exists but sidecar migration is pending 或等价语义
- **AND** 错误信息 MUST 包含可执行的预处理升级或强校验命令提示

#### Scenario: 不安全旧 sidecar 拒绝迁移
- **WHEN** 旧 sidecar 与当前 source path、source fingerprint、profile、split、`seq_len`、`num_pred`、shape、dtype 或 sample count 不匹配
- **THEN** 系统 MUST 拒绝 metadata-only upgrade
- **AND** `read_only` policy MUST 清晰失败
- **AND** `auto` policy MUST 记录 mismatch 原因并按现有安全策略重建 cache 或回退

#### Scenario: 强校验升级记录 fingerprint scan
- **WHEN** 用户显式请求 strong validation 迁移旧 sidecar
- **THEN** 系统 MUST 重新计算原始 source fingerprint 并与 sidecar 记录值比较
- **AND** 系统 MUST 在 sidecar 或预处理输出中记录 strong validation 耗时、结果和是否扫描 source fingerprint
- **AND** fingerprint 不匹配时系统 MUST 拒绝 metadata-only upgrade

### Requirement: Multimodal-NF 派生缓存迁移状态可追踪
Multimodal-NF cache status、runtime metadata 和预处理输出 MUST 能区分 valid cache、migration pending cache、invalid cache 和 missing cache。该状态 MUST 以机器可读字段暴露，便于 profile、推荐器和训练错误信息复用。

#### Scenario: cache status 暴露 migration pending
- **WHEN** 系统检查一个存在 `.npy` 数据文件且 sidecar 为可迁移旧 schema 的 Multimodal-NF image/LiDAR cache
- **THEN** cache status MUST 记录 `migration_pending=true` 或等价机器可读字段
- **AND** status MUST 记录 sidecar schema version、cache path、source path、validation mode 和待补齐字段摘要

#### Scenario: runtime metadata 记录升级行为
- **WHEN** dataset 构建过程中因 `auto` policy 执行 metadata-only sidecar upgrade
- **THEN** runtime metadata MUST 记录该模态发生 metadata upgrade
- **AND** metadata MUST 区分 `cache_generated=false`、`cache_rebuilt=false` 和 `metadata_upgraded=true` 或等价字段

#### Scenario: 预处理汇总迁移结果
- **WHEN** 用户运行 Multimodal-NF derived cache 预处理
- **THEN** 输出 MUST 按模态和 split 汇总 valid/skipped、metadata upgraded、rebuilt/generated、failed 和 missing 数量
- **AND** 输出 MUST 不包含真实大 cache 内容
