## ADDED Requirements

### Requirement: Baseline 源码与配置放置边界
系统 MUST 按行为而不是名称放置 baseline 相关实现。可由共享训练、评估、batch runtime 和模型架构摘要直接消费的模型能力 MUST 位于 `src/kd_sensing/models/` 或其窄子模块；论文复现、外部源码审计、多阶段训练、feature cache、特殊报告或 Table 风格 workflow MUST 位于 `src/kd_sensing/baselines/<family>/`、包内 CLI 或 package console script。本地可训练 baseline/control 配置 MUST 默认位于 `configs/fusion/` 或其实验子目录；外部复现、官方 artifact 审计或 source-audit manifest 配置 MUST 位于 `configs/baselines/`。

#### Scenario: 普通可训练 baseline 保持在模型组件路径
- **WHEN** baseline 能通过 `modular_sequence`、encoder/projector/representation core/head 或已有 whole-model exception 被共享训练 runtime 构建
- **THEN** 其模型实现 MUST 位于 `src/kd_sensing/models/` 或现有模型组件 owner
- **AND** 其本地训练配置 MUST 位于 `configs/fusion/`、`configs/fusion/experiments/` 或对应 current config family
- **AND** 系统 MUST 不因为名称包含 baseline 就把模型实现搬入 `src/kd_sensing/baselines/`

#### Scenario: Workflow baseline 使用 baseline package
- **WHEN** baseline 包含官方源码审计、多阶段训练、feature cache、专用 evaluation/report builder 或 Table 风格报告
- **THEN** workflow 实现 MUST 位于 `src/kd_sensing/baselines/<family>/`、包内 CLI 或 package console script
- **AND** 该 workflow MUST 不注册新的 `MODELS`、`ENCODERS`、`PROJECTORS`、`REPRESENTATION_CORES` 或 `HEADS` 名称来绕过模型组件边界

#### Scenario: 外部复现配置与本地训练配置分开
- **WHEN** 配置描述本仓库可训练 baseline/control
- **THEN** 配置 MUST 使用 `configs/fusion/` 或 current experiment config family
- **AND** 当配置描述外部 repo、官方 checkpoint、官方 prediction、source audit 或 blocked official reproduction 时，配置 MUST 使用 `configs/baselines/` 或明确的 diagnostics manifest 路径
