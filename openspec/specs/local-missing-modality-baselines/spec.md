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
