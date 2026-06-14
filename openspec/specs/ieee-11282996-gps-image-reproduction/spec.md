# ieee-11282996-gps-image-reproduction Specification

## Purpose
定义 AMR-Net_gps_image / IEEE `11282996` source audit、GPS+Image-only local substitute、Scenario 23 边界、paper-aligned metrics、报告产物和 official reproduction claim gating，防止缺少官方协议证据时误声明复现结果。
## Requirements
### Requirement: AMR-Net_gps_image source audit
系统 MUST 在运行真实复现前生成 AMR-Net_gps_image / IEEE `11282996` source audit。Audit MUST 记录 IEEE URL、article number、title、DOI、venue/year、PDF availability、作者或官方代码 URL/commit、数据场景、split、启用模态、target label、metric profile、官方权重状态、阻塞项和 audit digest。

#### Scenario: IEEE metadata 可用
- **WHEN** 用户提供 IEEE PDF、BibTeX、作者页面或可访问 metadata
- **THEN** source audit MUST 记录 title、DOI、venue/year、article number 和来源 URL
- **AND** report MUST 标记 `source_audit.status` 为 `complete` 或 `paper_protocol_audited`

#### Scenario: IEEE 页面不可访问
- **WHEN** IEEE Xplore URL 无法从当前环境加载且用户未提供 PDF 或 metadata
- **THEN** source audit MUST 记录 blocked reason
- **AND** runner MUST 禁止将后续结果标记为 `official_reproduction`

#### Scenario: IEEE article metadata 与 Scenario 23 作者包冲突
- **WHEN** 公开 IEEE/Crossref metadata 将 article `11282996` 解析为非 Scenario 23 GPS+Image drone paper
- **THEN** source audit MUST 记录 `article_metadata_conflict: true`
- **AND** report MUST 同时记录冲突 article metadata 与 Scenario 23 local substitute metadata
- **AND** runner MUST 禁止将 Scenario 23 local substitute 标记为 `official_reproduction`

### Requirement: GPS+Image-only modality boundary
AMR-Net_gps_image 复现入口 MUST 只启用 canonical `image` 和 `gps` 模态。该入口 MUST 拒绝 LiDAR、radar、mmWave、CSI、all-modalities config、GPS+LiDAR BGAM checkpoint 或任何需要未启用模态输入的 fallback。

#### Scenario: paper config 只启用 image 和 gps
- **WHEN** 用户加载 AMR-Net_gps_image paper preset
- **THEN** 标准化模态列表 MUST 等于 `["image", "gps"]`
- **AND** dataset flag MUST 只启用 image 输入和 `use_gps: true`

#### Scenario: LiDAR 被拒绝
- **WHEN** 用户在 AMR-Net_gps_image preset、manifest 或 CLI override 中配置 `lidar`、`use_lidar: true` 或 GPS+LiDAR BGAM checkpoint
- **THEN** 系统 MUST 拒绝运行
- **AND** 错误信息 MUST 说明该复现只允许 GPS 与 image 模态

### Requirement: Paper protocol model groups
系统 MUST 为 AMR-Net_gps_image 复现声明 paper protocol model groups。模型组 MUST 至少能表达 image-only、GPS-only 和 Image+GPS fusion；若 source audit 证明论文未报告某组，该组 MUST 标记为 `local_control`，不得作为论文目标行。

#### Scenario: 构建 image-only model group
- **WHEN** manifest 声明 image-only paper model
- **THEN** 模型 MUST 只消费 image batch
- **AND** runtime metadata MUST 记录该模型不消费 GPS 或 LiDAR

#### Scenario: 构建 GPS-only model group
- **WHEN** manifest 声明 GPS-only paper model
- **THEN** 模型 MUST 只消费 GPS batch
- **AND** runtime metadata MUST 记录 GPS feature mode、normalizer provenance 和 target source

#### Scenario: 构建 Image+GPS fusion model group
- **WHEN** manifest 声明 Image+GPS fusion paper model
- **THEN** 模型 MUST 消费 image 与 GPS batch 并输出 beam classifier logits
- **AND** metadata MUST 记录 image encoder、GPS encoder、fusion type、num beams 和是否为 paper-reported row

### Requirement: Paper-aligned metrics and report
AMR-Net_gps_image 复现 MUST 输出 paper-aligned metrics 和可审计 report。Metrics MUST 至少覆盖 Top-1、Top-3 和 Top-5 beam accuracy；若 paper 或本地 metric helper 支持 DBA、beam-distance、overhead reduction 或 normalized gain，输出 MUST 使用明确字段名并记录口径。

#### Scenario: 写出 metrics summary
- **WHEN** 复现评估完成
- **THEN** 输出目录 MUST 包含 machine-readable metrics summary
- **AND** summary MUST 记录 model group、scene、split、sample_count、Top-1、Top-3、Top-5、metric profile、seed 和 claim status

#### Scenario: 写出复现 report
- **WHEN** runner 完成真实、mock 或 blocked 运行
- **THEN** 输出目录 MUST 包含 report 或 manifest
- **AND** report MUST 记录 source audit digest、命令、git status 摘要、enabled modalities、checkpoint provenance、dataset path、warnings 和生成文件清单

### Requirement: Claim status gating
系统 MUST 用 claim status 区分官方复现、本地替代、local control、mock smoke 和 blocked 状态。没有完整 source audit、官方数据/split、官方权重或 exact training/evaluation protocol 时，系统 MUST NOT 声称 official reproduction。

#### Scenario: official 条件不完整
- **WHEN** source audit 缺少 PDF、官方代码、官方 split、官方权重或 exact protocol 中任一关键项
- **THEN** claim status MUST 为 `blocked_official`、`paper_protocol_audited`、`local_substitute`、`local_control` 或 `mock_smoke`
- **AND** report MUST 明确列出缺失项

#### Scenario: article conflict 禁止 official claim
- **WHEN** source audit 标记 `article_metadata_conflict: true`
- **THEN** claim status MUST NOT be `official_reproduction`
- **AND** report MUST include the conflicting DOI/title/evidence URLs that caused the block

#### Scenario: official 条件完整
- **WHEN** source audit、数据、split、权重、训练/评估协议和 metric profile 均与论文一致
- **THEN** runner MAY 标记 `official_reproduction`
- **AND** report MUST 记录每个 official 条件的证据路径或 digest

### Requirement: Runtime artifact boundary
AMR-Net_gps_image 复现生成的 metrics、predictions、cache、checkpoint、plots 和 reports MUST 写入 ignored 的 runtime output root。源码变更 MUST NOT 要求提交真实数据、运行产物、checkpoint 或下载的 IEEE PDF。

#### Scenario: 输出写入 ignored root
- **WHEN** runner 生成复现产物
- **THEN** 默认输出路径 MUST 位于 `outputs/analysis/amr_net_gps_image/`、scene-scoped `outputs/scene23/` 或用户显式指定的 ignored 输出目录
- **AND** report MUST 记录输出 root 和文件清单

#### Scenario: mock 结果不可冒充真实结果
- **WHEN** runner 使用 mock 或 synthetic 数据
- **THEN** metrics、checkpoint metadata 和 report MUST 标记 `mock_data: true`
- **AND** claim status MUST 为 `mock_smoke`
