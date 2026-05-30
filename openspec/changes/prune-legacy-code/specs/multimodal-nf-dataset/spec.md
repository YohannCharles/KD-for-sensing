## REMOVED Requirements

### Requirement: Multimodal-NF 本地数据布局与审计
**Reason**: Multimodal-NF 数据集家族退役，审计、默认目录和 HDF5/codebook 检查不再维护。
**Migration**: 使用当前保留的数据集家族；本地已有 `dataset/MultimodalNF` 可作为静态文件保留，但项目不再读取或审计它。

#### Scenario: Multimodal-NF 审计入口删除
- **WHEN** 用户运行 Multimodal-NF 审计配置
- **THEN** 系统 MUST 因配置不存在或 preprocessor 不可用而失败
- **AND** 项目文档 MUST 不再推荐该入口

### Requirement: Multimodal-NF HDF5 index 构建
**Reason**: Multimodal-NF frame-wise index builder 随数据集支持一起退役。
**Migration**: HDF5/index 类数据集若未来需要支持，应重新定义 dataset runtime capability。

#### Scenario: index builder 不再提供
- **WHEN** 用户查找 Multimodal-NF index 构建配置或 preprocessor
- **THEN** 系统 MUST 不再提供 `multimodal_nf_index` 工作流

### Requirement: Multimodal-NF dataset sample 契约
**Reason**: `multimodal_nf` dataset type 已删除，flat sample 契约不再适用。
**Migration**: 使用保留数据集的 sample contract。

#### Scenario: dataset type 不可用
- **WHEN** 用户配置 `data.dataset.type: multimodal_nf`
- **THEN** dataset 构建 MUST 失败
- **AND** 错误信息 MUST 指出该 dataset type 不受支持

### Requirement: Multimodal-NF 近场 beam target 契约
**Reason**: 近场三维 codebook target 只服务于退役 Multimodal-NF。
**Migration**: 当前 beam 预测目标继续使用保留数据集自己的 target schema。

#### Scenario: target_beam codebook 契约删除
- **WHEN** 当前保留数据集构建样本
- **THEN** 系统 MUST 不要求输出 Multimodal-NF `beam_triplet_topk` 或 `beam_power_topk`

### Requirement: Multimodal-NF 辅助标签契约
**Reason**: LoS/NF/trajectory mode 辅助标签契约绑定退役数据集。
**Migration**: 其它数据集的辅助标签由各自 capability 约束。

#### Scenario: 辅助标签不再作为 Multimodal-NF 要求
- **WHEN** 训练运行写出 metadata
- **THEN** 系统 MUST 不要求记录 Multimodal-NF 辅助标签可用性

### Requirement: Multimodal-NF 配置和 smoke workflow
**Reason**: Multimodal-NF 配置、fixture 和 smoke workflow 已退役。
**Migration**: 使用当前保留数据集的 smoke test。

#### Scenario: 配置目录删除
- **WHEN** 开发者检查源码配置目录
- **THEN** `configs/multimodal_nf/` MUST 不再作为支持配置目录存在
- **AND** `tests/test_multimodal_nf_*` 正向测试 MUST 被删除

### Requirement: Multimodal-NF helper 拆分兼容
**Reason**: helper 拆分兼容只服务于退役 Multimodal-NF 实现。
**Migration**: 不提供兼容；相关公开 preprocessor registry 名称应删除。

#### Scenario: helper 兼容不再维护
- **WHEN** 用户引用旧 Multimodal-NF helper 或 preprocessor 名称
- **THEN** 系统 MUST 不再保证导入或构建成功

### Requirement: Multimodal-NF capability purpose 明确
**Reason**: 整个 capability 已退役，不再需要维护目的说明。
**Migration**: archive 后该 capability 可从 active specs 中移除。

#### Scenario: active specs 不含 Multimodal-NF capability
- **WHEN** change 归档完成后开发者查看 active specs
- **THEN** 系统 MUST 不再包含 `multimodal-nf-dataset` active capability

### Requirement: Multimodal-NF objective runtime 语义
**Reason**: Multimodal-NF objective runtime metadata 随数据集退役。
**Migration**: 当前保留 objectives 继续记录各自 runtime metadata。

#### Scenario: runtime metadata 不再记录 Multimodal-NF objective
- **WHEN** 当前保留训练配置启动
- **THEN** runtime metadata MUST 不包含 `dataset_type: multimodal_nf`

### Requirement: Multimodal-NF codebook consistency
**Reason**: codebook shape 与 beam head 一致性检查只服务于退役近场 codebook 目标。
**Migration**: 保留数据集的类别数校验继续由其任务契约约束。

#### Scenario: codebook 校验入口删除
- **WHEN** 配置包含 Multimodal-NF codebook metadata
- **THEN** 系统 MUST 不把该字段作为受支持配置处理

### Requirement: Multimodal-NF image/LiDAR 派生缓存
**Reason**: Multimodal-NF image/LiDAR 派生缓存随数据集退役。
**Migration**: 不迁移缓存；用户本地缓存可自行删除或保留为静态文件。

#### Scenario: 派生缓存不再生成
- **WHEN** 用户运行当前保留预处理入口
- **THEN** 系统 MUST 不生成 Multimodal-NF image/LiDAR 派生缓存

### Requirement: Multimodal-NF 派生缓存审计与可追踪性
**Reason**: 派生缓存审计只服务于退役缓存格式。
**Migration**: 当前保留数据集的 cache metadata 由对应 specs 约束。

#### Scenario: 缓存审计字段不再要求
- **WHEN** profile 或 runtime metadata 写出
- **THEN** 系统 MUST 不要求包含 Multimodal-NF cache coverage 或 source_kind 字段

### Requirement: Multimodal-NF 派生缓存轻量校验
**Reason**: 轻量 sidecar 校验只服务于退役派生缓存。
**Migration**: 不提供迁移；如需读取历史缓存，应在外部脚本自行处理。

#### Scenario: read_only cache policy 不再支持
- **WHEN** 配置包含 `data.cache.multimodal_nf`
- **THEN** 系统 MUST 拒绝该配置或忽略为未知字段并给出清晰错误

### Requirement: Multimodal-NF 派生缓存 IO 布局元数据
**Reason**: cache IO 布局元数据只服务于退役缓存。
**Migration**: 无迁移。

#### Scenario: sidecar schema 不再维护
- **WHEN** 开发者运行测试
- **THEN** 测试 MUST 不再校验 Multimodal-NF sidecar storage kind、layout 或 shard 字段

### Requirement: Multimodal-NF 派生缓存 lazy 读取边界
**Reason**: lazy cache adapter 随 Multimodal-NF dataset 删除。
**Migration**: 当前保留数据集的 lazy loading 由各自 specs 约束。

#### Scenario: lazy adapter 删除
- **WHEN** 代码扫描 dataset adapters
- **THEN** Multimodal-NF image/LiDAR cache adapter MUST 不再作为支持路径存在

### Requirement: Multimodal-NF 旧派生缓存 sidecar 迁移
**Reason**: 历史 sidecar 迁移只服务于退役缓存格式。
**Migration**: 不迁移历史 sidecar；本地文件可静态保留。

#### Scenario: sidecar 迁移命令删除
- **WHEN** 用户查找 Multimodal-NF cache sidecar migration 命令
- **THEN** 系统 MUST 不再提供该命令或配置

### Requirement: Multimodal-NF 派生缓存迁移状态可追踪
**Reason**: cache migration 状态追踪随 sidecar 迁移退役。
**Migration**: 无迁移。

#### Scenario: migration 状态不再输出
- **WHEN** 训练吞吐 profile 或 run metadata 写出
- **THEN** 系统 MUST 不要求包含 Multimodal-NF cache migration pending/done 字段
