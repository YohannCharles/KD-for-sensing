# retired-route-summary Specification

## Purpose
集中记录已折叠退役路线的拒绝边界、历史语义和迁移方向。该能力是 project surface guard，不是当前训练、评估、诊断或数据准备入口。
## Requirements
### Requirement: 折叠退役路线不属于 current support surface
Beam distribution shift、BeamBench、BEV-Fusion 2604、CxD/Scenario-D、dataset audit、DeepSense legacy LOSO helper、geometry prior/safe rerank、GPS-query pooling/evidence、JEPA shortcut/visual/predictive suites、legacy local baseline pack/FeatureMod、modality visual diagnostics、standalone model-architecture CLI/renderer/sweep、旧独立 RBMA/prototype-KD sweep/config/runbook、real perturbation benchmark、Scene31 legacy summary/baseline/next-round/subset、standalone target-shot CLI、TII、throughput、Vision-Position 和 WCL source-audit MUST 只作为 retired、historical 或 migration guard 出现。项目 MUST 不为这些路线提供 current console script、module-only CLI、实体训练 YAML、registry current entry、package facade 或 local runbook。MMW 复用的 target-shot split helper、训练/Scene31-34 复用的 instance/startup architecture summary、Scene31-34 使用的 AMR-lite，以及 U-Mask 内嵌 RBMA/prototype/full-to-partial teacher 分支 MUST 继续作为 current/supporting owner。

#### Scenario: 旧入口不会回流
- **WHEN** 开发者检查 pyproject、CLI、scripts、configs 和 current docs
- **THEN** 上述退役路线 MUST 不作为 current 推荐入口出现
- **AND** 文本提到旧名称时 MUST 明确 retired、historical、removed 或 compatibility 语境

#### Scenario: 旧配置和 module 快速拒绝
- **WHEN** 用户传入已退役 config path、override、model type 或 module command
- **THEN** 配置加载、registry 或入口检查 MUST fail fast 或返回普通 unknown-name 错误
- **AND** 系统 MUST 不静默迁移到 current workflow

#### Scenario: 通用能力仍由 current owner 使用
- **WHEN** current workflow 需要 circular metric、Top-K、label-space、artifact reader、JEPA mean pooling 或 MMW split helper
- **THEN** 实现 MUST 从 current owner 使用该通用能力
- **AND** 复用 MUST 不恢复退役 workflow 名称或专属入口

#### Scenario: Supporting owner 不被 retired token 误删
- **WHEN** MMW protocol 调用 target-shot split helper，或 startup/Scene31-34 调用 instance architecture summary
- **THEN** 对应 helper 和行为契约 MUST 保留
- **AND** project surface MUST 仍拒绝其已退役 standalone CLI、renderer 或 sweep 入口

#### Scenario: RBMA 名称按 owner 区分
- **WHEN** cleanup 扫描 RBMA、prototype 或 KD token
- **THEN** 它 MUST 只删除旧独立 sweep/config/runbook 与 retired workflow naming
- **AND** `u_mask_beam_jepa` 的 `reliability_biased_missing_attention`、prototype alignment、full-to-partial teacher stabilization 和 pattern metrics MUST 保留

### Requirement: 集中 retired-route guard 取代专用 tombstone 测试
退役路线防回流 MUST 由集中 token/path 清单、参数化测试和普通 unknown-name 行为维护。项目 MUST 不为每条只剩历史说明的路线保留独立 current spec、专用 pytest、兼容 wrapper 或隐藏 CLI。

#### Scenario: 集中测试覆盖拒绝点
- **WHEN** 运行 retired-route focused tests
- **THEN** 测试 MUST 覆盖代表性旧 config、console command、module path、registry token 和 current docs 推荐语境
- **AND** 单条路线 MUST 不再拥有只验证“不存在”的专属测试文件

#### Scenario: 历史细节仍可查询
- **WHEN** 维护者需要理解已折叠能力的完整历史需求
- **THEN** 文档 MUST 指向 dated OpenSpec archive 或 git history
- **AND** current specs MUST 不复制完整历史 requirements
