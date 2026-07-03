# local-missing-modality-baselines Specification

## Purpose
约束 AMBER-lite、RMBP-MM 和 TII-VLRG-style 缺失模态 baseline 在本仓库实验场景中的本地训练、验证、配置、metadata 和 claim 边界。
## Requirements
### Requirement: 本地可训练 baseline 入口
系统 MUST 为 AMBER-lite、RMBP-MM 和 TII-VLRG-style baseline 提供本仓库内可训练配置。默认训练入口 MUST 使用 `kd-sensing-train --config`，且 MUST NOT 依赖官方源码、外部 checkpoint 或自动下载权重。

#### Scenario: 使用本地配置训练
- **WHEN** 用户选择任一 local missing-modality baseline 配置
- **THEN** 系统 MUST 能通过 `conda run -n kd_mm_beam kd-sensing-train --config <config>` 构建模型并进入训练/验证流程
- **AND** 默认配置 MUST 使用本地 scratch 权重或用户显式提供的本地 checkpoint

#### Scenario: external wrapper 不作为训练前置
- **WHEN** TII 或 WCL external/audit artifact 缺失
- **THEN** 本地 baseline 训练配置 MUST 仍可独立构建
- **AND** 系统 MUST NOT 将 external wrapper 缺失解释为本地 baseline 不可用

### Requirement: 缺失模态训练扰动接入
系统 MUST 使用现有 difficulty pipeline 表达训练期缺失模态扰动，不得保留训练循环不消费的伪配置字段作为唯一 dropout 声明。

#### Scenario: 训练期 dropout 生效
- **WHEN** baseline 配置声明 missing-modality train profile
- **THEN** `BatchStepRunner` 的训练 batch MUST 通过 `apply_configured_difficulty` 应用该 profile
- **AND** 输入 mask metadata MUST 能被 opt-in 模型消费或由 zero-imputation baseline 显式记录

### Requirement: baseline claim 边界
系统 MUST 将这些条目标记为 local experimental baseline。文档、metadata 和 summary MUST NOT 声称 official reproduction 或论文数值复现。

#### Scenario: 输出和文档标记为 local baseline
- **WHEN** baseline 配置、文档或 summary 描述 AMBER-lite、RMBP-MM 或 TII-VLRG-style 条目
- **THEN** 它们 MUST 标记为 local experimental baseline
- **AND** official/external/audit wrapper MUST 仅作为可选参考路径出现

### Requirement: AMBER full 纳入本地缺失模态 baseline 家族
系统 MUST 将 AMBER full architecture reproduction 作为 local missing-modality baseline 家族中的本地架构复现条目。该条目 MUST 使用当前训练入口和现有 missing/difficulty metadata 边界，不得依赖官方源码、外部 checkpoint 或专用训练循环。

#### Scenario: 使用本地训练入口启动 AMBER full
- **WHEN** 用户选择 AMBER full architecture config
- **THEN** 系统 MUST 能通过 `conda run -n kd_mm_beam kd-sensing-train --config <config>` 构建模型并进入训练/验证流程
- **AND** 默认配置 MUST 使用本地 scratch 权重或用户显式提供的本地 checkpoint

#### Scenario: AMBER full claim 保持本地实验状态
- **WHEN** 文档、summary 或 claim row 描述 AMBER full
- **THEN** 它 MUST 标记为 local architecture reproduction 或 local experimental baseline
- **AND** 缺少严格可比真实评估时 MUST NOT 声称 official AMBER reproduction

### Requirement: Local baseline stress comparability
本地缺失模态 baseline MUST 能声明 stress benchmark comparability metadata。适用对象包括 AMBER-lite、AMBER full、RMBP-MM、U-MaskBeamJEPA/weighted_sum 和其它 current local missing-modality baseline。

#### Scenario: baseline 声明 required fields
- **WHEN** baseline 被纳入 missing-modality stress suite
- **THEN** baseline manifest 或 run metadata MUST 声明 config path、weights path、checkpoint provenance、modalities、split、sample_count、label_space、metric_profile、target_source、seed 和 difficulty_digest
- **AND** 缺失 required field MUST 阻止该 baseline 进入 strict claim comparison

#### Scenario: local substitute 状态保留
- **WHEN** AMBER、RMBP-MM 或其它外部论文 baseline 使用本仓库 local implementation
- **THEN** stress summary MUST 保留 `local experimental baseline` 或 `local substitute` 状态
- **AND** 系统 MUST 不将其描述为 official reproduction

#### Scenario: baseline 缺某模态
- **WHEN** stress suite 包含 baseline 不支持的模态或 condition
- **THEN** 对应 row MUST 标记为 unavailable、not_applicable 或 not_comparable
- **AND** summary MUST 不把缺失 row 当作 0 分或 clean 结果

