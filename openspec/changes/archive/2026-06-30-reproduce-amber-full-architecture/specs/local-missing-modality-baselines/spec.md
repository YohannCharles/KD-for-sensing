## ADDED Requirements

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
