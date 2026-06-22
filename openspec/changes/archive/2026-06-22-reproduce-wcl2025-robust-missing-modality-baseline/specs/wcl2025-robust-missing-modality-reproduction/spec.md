## ADDED Requirements

### Requirement: WCL 2025 source audit
系统 MUST 为 IEEE WCL 2025 missing-modality baseline 生成 source-audit manifest。manifest MUST 记录论文、代码、权重、数据集、模态、split、metric 和训练 recipe 的可用性。

#### Scenario: source audit 完成
- **WHEN** 用户运行 WCL 2025 source-audit 或 reproduction dry-run
- **THEN** 系统 MUST 写出 source-audit manifest
- **AND** manifest MUST 至少包含 paper title、citation metadata、code URL、source commit、license/availability、checkpoint availability、dataset、enabled modalities、split、metric profile 和 missing details

#### Scenario: 官方 artifact 不可用
- **WHEN** 官方代码、权重、训练 recipe 或必要数据协议不可用
- **THEN** manifest MUST 将 official reproduction status 标记为 blocked、pending 或 unavailable
- **AND** 系统 MUST NOT 声称 official reproduction

### Requirement: Official 与 local-substitute 分支
WCL 2025 reproduction MUST 区分 official-code reproduction 和 paper-aligned local substitute。两个分支 MUST 使用统一 summary schema，但 claim status MUST 明确不同。

#### Scenario: official-code reproduction
- **WHEN** 官方代码、权重、数据协议和 metric 口径均可用
- **THEN** 系统 MAY 包装官方运行或导入官方预测/metrics
- **AND** summary MUST 标记 `claim_status: official_reproduction` 或等价状态，并记录 source commit 和 checkpoint provenance

#### Scenario: local substitute reproduction
- **WHEN** 官方 artifact 缺失但论文结构可由本仓库实现
- **THEN** 系统 MAY 实现 paper-aligned local substitute
- **AND** summary MUST 标记 `claim_status: local_substitute`
- **AND** manifest MUST 记录与论文结构、训练流程、数据集或 metric 的 deviation

### Requirement: WCL 2025 missing-modality model
WCL 2025 local substitute MUST 支持论文对齐的缺失模态 beam prediction 结构。实现 MUST 优先使用可组合 encoder/projector/core/head 或窄 workflow owner。

#### Scenario: 构建 local substitute 模型
- **WHEN** 配置声明 WCL 2025 local substitute
- **THEN** 系统 MUST 构建论文对齐的 per-modality encoder 和 missing-modality fusion 结构
- **AND** metadata MUST 记录 enabled modalities、missing-modality strategy、fusion type、paper alignment 和 deviation

#### Scenario: whole-model exception 需要理由
- **WHEN** WCL 2025 结构无法表达为可组合组件并需要新增完整模型
- **THEN** design 或 implementation note MUST 说明 whole-model exception 理由
- **AND** tasks MUST 覆盖 registry build、synthetic forward、ModelOutput adaptation 和 metadata tests

### Requirement: WCL 2025 condition-level evaluation
WCL 2025 reproduction MUST 输出 clean 和 missing-modality condition-level metrics，并保留 strict comparability metadata。

#### Scenario: 输出 condition metrics
- **WHEN** official 或 local-substitute evaluation 完成
- **THEN** 系统 MUST 写出 clean、单模态缺失、多模态缺失和论文声明关键 missing conditions 的 Top-K、DBA 或 beam distance metrics
- **AND** 每行 MUST 包含 condition id、affected modalities、sample count、split、seed、metric profile 和 provenance

#### Scenario: strict mismatch 不进入 ranking
- **WHEN** WCL 2025 row 的 split、scene set、label space、metric profile、sample count、seed 或 difficulty digest 与当前 strict protocol 不一致
- **THEN** 系统 MUST 将 row 标记为 not_comparable 或 external_reference
- **AND** 该 row MUST NOT 进入 strict ranking 或 claim upgrade

### Requirement: WCL 2025 产物边界
WCL 2025 reproduction 的外部源码、checkpoint、cache、prediction、metrics、logs 和 figures MUST 位于 ignored runtime output root 或用户显式指定路径。

#### Scenario: 运行产物不进入源码
- **WHEN** WCL 2025 reproduction 生成下载源码、checkpoint、cache、log、prediction、metrics 或 figure
- **THEN** 这些产物 MUST 位于 ignored `outputs/analysis/wcl2025_missing_modality_reproduction/` 或用户显式指定 output root
- **AND** 源码变更 MUST 只包含代码、配置、测试、OpenSpec 和文档

#### Scenario: tests 不依赖真实 artifact
- **WHEN** focused tests 验证 WCL 2025 source audit、branch selection、模型或 summary adapter
- **THEN** tests MUST 使用 synthetic manifest、synthetic tensors 或 fixture metrics
- **AND** tests MUST NOT 读取真实 `dataset/`、下载外部 repo 或加载 checkpoint
