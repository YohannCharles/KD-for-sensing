## ADDED Requirements

### Requirement: AMBER-lite baseline model
系统 MUST 提供 AMBER-lite missing-modality baseline，用于融合 image、LiDAR、radar 和 GPS 模态并在模态缺失时使用 mask token 或等价缺失表征。

#### Scenario: 构建 AMBER-lite 配置
- **WHEN** 配置声明 AMBER-lite baseline
- **THEN** 系统 MUST 能构建 image、LiDAR、radar 和 GPS encoder/projector 组合及 missing-modality fusion core
- **AND** 模型 metadata MUST 记录启用模态、fusion core、是否消费 missing metadata 和 mask token 策略

#### Scenario: 普通 baseline 不需要 missing metadata
- **WHEN** 非 AMBER-lite baseline 运行
- **THEN** missing-modality metadata MUST NOT 成为必需 forward 输入
- **AND** 现有 Image+GPS、GPS-only、JEPA 和 CNN baseline MUST 保持可构建和可 forward

### Requirement: Modality dropout 训练 profile
AMBER-lite MUST 支持训练期 modality dropout profile。该 profile MUST 只影响输入模态字段和输入 mask metadata，不得改变 target beam、beam power、sample id、split 或 target schema。

#### Scenario: 训练时随机 drop 模态
- **WHEN** 训练配置启用 AMBER-lite modality dropout
- **THEN** dataloader、batch transform 或模型输入准备 MUST 按配置随机 drop image、LiDAR、radar 或 GPS
- **AND** run metadata MUST 记录 dropout rates、affected modalities、seed、digest 和 fallback count

#### Scenario: dropout 不移动 target
- **WHEN** modality dropout 应用于 train batch
- **THEN** target_beam、beam_power、auxiliary target、sample id 和 split metadata MUST 与 clean batch 保持一致
- **AND** metadata MUST 标记扰动作用于输入模态而非 target schema

### Requirement: Missing-modality evaluation
AMBER-lite reproduction MUST 支持 clean 和 missing/degraded condition-level evaluation。condition MUST 能覆盖单模态缺失、多模态缺失、poor image、LiDAR/radar unavailable、wrong/async GPS 和组合缺失。

#### Scenario: 解析 evaluation conditions
- **WHEN** evaluation manifest 声明 AMBER-lite missing-modality suite
- **THEN** 系统 MUST 标准化每个 condition 的 id、affected modalities、operator params、seed、split、difficulty digest 和 expected availability mask
- **AND** 未启用的 condition MUST 不影响 clean evaluation

#### Scenario: condition-level metrics 输出
- **WHEN** AMBER-lite evaluation 完成或导入 metrics
- **THEN** 系统 MUST 写出 clean 和每个 condition 的 Top-K、DBA 或 beam distance metrics
- **AND** output row MUST 包含 strict comparability fields、dropout policy、missing-mask provenance 和 status

### Requirement: AMBER-lite claim 边界
AMBER-lite reproduction MUST 明确标记为 local lite reproduction。缺少 strict comparable real run、缺少 LiDAR/radar 数据、使用 synthetic metrics 或只完成 dry-run 时，系统 MUST 不升级为真实性能 claim。

#### Scenario: lite reproduction 标记
- **WHEN** AMBER-lite summary、manifest 或文档被生成
- **THEN** provenance MUST 包含 `reproduction_scope: amber_lite_local`
- **AND** report MUST NOT 声称完整 AMBER 官方复现

#### Scenario: unavailable 不进入 strict ranking
- **WHEN** LiDAR/radar artifact、checkpoint、metrics 或 strict comparability 字段缺失
- **THEN** 系统 MUST 将 row 标记为 pending、unavailable 或 not_comparable
- **AND** 该 row MUST NOT 进入 strict ranking 或 claim upgrade

### Requirement: AMBER-lite 产物边界
AMBER-lite 训练、评估、checkpoint、log、mask diagnostics、metrics、prediction 和图表 MUST 写入 ignored runtime output root。

#### Scenario: 运行产物不进入源码
- **WHEN** AMBER-lite baseline 生成本地训练或评估产物
- **THEN** checkpoint、cache、log、metrics、prediction 和 figures MUST 位于 ignored `outputs/analysis/amber_lite_missing_modality/` 或用户显式指定 output root
- **AND** 源码变更 MUST 只包含代码、配置、测试、OpenSpec 和文档

#### Scenario: tests 使用 synthetic inputs
- **WHEN** focused tests 验证 AMBER-lite 模型、dropout 或 summary adapter
- **THEN** tests MUST 使用 synthetic tensors、small fixture rows 或 dry-run manifest
- **AND** tests MUST NOT 读取真实 `dataset/`、checkpoint 或本地运行产物
