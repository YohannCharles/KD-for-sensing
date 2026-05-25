## ADDED Requirements

### Requirement: Multimodal-NF 派生缓存轻量校验
Multimodal-NF image/LiDAR 派生缓存 MUST 支持轻量运行时校验和显式强校验。默认训练或 `read_only` cache 读取路径 MUST 不在每次 dataset 构建时重新扫描原始 HDF5 大文件计算 fingerprint；强 fingerprint 校验 MUST 通过显式预处理、审计、重建或配置选项触发，并 MUST 在 metadata 中记录结果。

#### Scenario: read_only cache 初始化不扫描原始大文件
- **WHEN** 用户配置 `data.cache.multimodal_nf.image.policy=read_only` 或 `data.cache.multimodal_nf.lidar.policy=read_only`，且对应 sidecar 的轻量校验字段有效
- **THEN** dataset 初始化 MUST 使用 sidecar 中的路径、大小、mtime、profile、split、`seq_len`、`num_pred`、shape、dtype 和 cache version 进行轻量校验
- **AND** dataset 初始化 MUST NOT 重新读取整个原始 HDF5 文件计算 SHA256 fingerprint
- **AND** runtime metadata MUST 记录 validation mode、是否执行 source fingerprint scan 和校验耗时

#### Scenario: 显式强校验
- **WHEN** 用户通过预处理、审计或配置显式请求 Multimodal-NF 派生缓存强校验
- **THEN** 系统 MUST 重新计算原始源文件 fingerprint 并与 sidecar 记录值比较
- **AND** 系统 MUST 在校验报告或 runtime metadata 中记录强校验耗时、结果和不匹配原因
- **AND** 若 policy 为 `read_only` 且强校验不匹配，系统 MUST 拒绝读取该 cache 并输出包含 cache path、source path 和 mismatch 字段的错误

#### Scenario: 旧 sidecar 缺少轻量校验字段
- **WHEN** sidecar 缺少轻量校验必需字段或 cache version 不可识别
- **THEN** `read_only` policy MUST 给出清晰错误，提示用户执行强校验、重建或切换 policy
- **AND** `auto` policy MAY 按现有策略重建或回退到原始 HDF5，但 MUST 在 metadata 中记录 fallback/rebuild 原因

### Requirement: Multimodal-NF 派生缓存 IO 布局元数据
Multimodal-NF image/LiDAR 派生缓存 sidecar MUST 记录足以诊断和优化随机窗口读取的 IO 布局信息。dataset 和 profile MUST 能从 sidecar/runtime metadata 中报告 cache 的 storage kind、layout、分片或 source key、样本数、字节数、shape、dtype 和推荐访问模式。

#### Scenario: sidecar 记录 IO 布局
- **WHEN** 系统生成或重建 Multimodal-NF image/LiDAR 派生缓存
- **THEN** sidecar MUST 记录 `storage_kind`、`layout`、`source_key` 或等价 shard 标识、`sample_count`、`bytes`、shape、dtype、cache version 和推荐访问模式
- **AND** sidecar MUST 继续记录原始 source path 或等价 source identity，便于审计和回退

#### Scenario: runtime metadata 暴露 cache 读取计划
- **WHEN** dataset 使用 Multimodal-NF image/LiDAR 派生缓存构建 train/test split
- **THEN** runtime metadata MUST 记录每个启用模态的 source kind、cache policy、validation mode、cache path 数量、总字节数和是否可能产生随机读风险
- **AND** 未启用 image/LiDAR 的配置 MUST 不解析、不校验、不打开对应派生 cache

#### Scenario: 派生 cache 样本等价
- **WHEN** 同一样本可从原始 HDF5 路径和派生 cache 路径读取
- **THEN** 派生 cache 路径 MUST 返回与原始路径等价的 sample keys、tensor shape、dtype 语义和 target 字段
- **AND** 任何新的 shard 或布局格式 MUST 保持该等价契约并提供可回退到原始 HDF5 的策略

### Requirement: Multimodal-NF 派生缓存 lazy 读取边界
Multimodal-NF dataset adapter MUST 以 worker-local lazy 方式打开派生 cache，并 MUST 避免在 worker 初始化或首次样本读取前 eager mmap 所有 city/source cache 文件。实现 MAY 提供打开文件数或映射字节数上限，但 MUST 保持读取结果与现有 sample 契约兼容。

#### Scenario: worker 只打开当前样本需要的 cache
- **WHEN** DataLoader worker 第一次读取某个 Multimodal-NF image/LiDAR 样本
- **THEN** adapter MUST 只打开该样本 source 对应的 cache 文件或 shard
- **AND** adapter MUST NOT 因启用该模态而立即 mmap 当前 split 下所有 cache 文件

#### Scenario: cache 打开状态可诊断
- **WHEN** profile 或 runtime metadata 请求 cache IO 诊断
- **THEN** 系统 MUST 能报告已打开 cache 文件数、映射字节数或等价计数
- **AND** 该诊断 MUST 不改变样本读取结果
