## REMOVED Requirements

### Requirement: Research claim harvesting
**Reason**: 自动扫描本地产物生成 claim candidate 的产品面没有核心调用方，并与人工 claim registry、run index 和 paper export 重叠。
**Migration**: 使用 `kd-sensing-runs` 定位产物，人工更新 `docs/result_claims_registry.md`，使用 paper export 输出审阅结果。

#### Scenario: Harvester 不再可用
- **WHEN** 用户请求旧 research dashboard/harvester 命令或 import
- **THEN** command/module MUST 不再属于 current surface
- **AND** current docs MUST 指向保留 owner

### Requirement: Strict comparability gate
**Reason**: Harvester 专属自动 gate 随 candidate pipeline 退役。
**Migration**: Comparability 由 experiment protocol、claim registry 和 paper export gate 人工审阅。

#### Scenario: Claim comparability 仍需记录
- **WHEN** claim 被加入正式 registry
- **THEN** registry/protocol MUST 记录 split、metric、seed 和 caveat
- **AND** 不要求 harvester gate

### Requirement: Experiment ledger
**Reason**: 独立 candidate ledger 与 run artifacts 和正式 claim registry 重复。
**Migration**: 保留 run metadata、run index 和正式 claim docs。

#### Scenario: Ledger 输出不再生成
- **WHEN** current workflow 完成 run
- **THEN** 系统 MUST 不要求 research ledger JSONL/CSV/SQLite
- **AND** 既有 ignored ledger MAY 作为本地产物保留

### Requirement: Daily research dashboard
**Reason**: Dashboard/JSON/HTML 展示面维护成本高且不是训练、评估或 claim 审阅必需路径。
**Migration**: 使用 run index、OpenSpec status、claim registry 和 paper export 的原生命令/文档。

#### Scenario: Dashboard CLI 删除
- **WHEN** 用户查看 pyproject entry points
- **THEN** `kd-sensing-research-dashboard` MUST 不存在
- **AND** 不提供替代 HTML dashboard wrapper

### Requirement: Claim doctor 输出缺失证据
**Reason**: 自动 next-action/upgrade candidate 逻辑属于退役 dashboard 产品面。
**Migration**: Claim 缺口由正式 registry caveat 与 experiment protocol checklist 记录。

#### Scenario: Pending claim 保持人工审阅
- **WHEN** claim 缺少证据
- **THEN** registry MUST 保持 pending/unverified/not-comparable 状态
- **AND** 系统不要求 claim doctor 输出

### Requirement: Run card artifact
**Reason**: Dashboard 专属 run card 写出不再保留；run index 已提供必要只读 provenance 字段。
**Migration**: 直接使用 run index JSON 和 run 目录中的 config/status/metrics metadata。

#### Scenario: Run provenance 可查询
- **WHEN** 维护者需要 run provenance
- **THEN** run index 与现有 run artifacts MUST 提供路径和状态
- **AND** 不要求独立 run card renderer
