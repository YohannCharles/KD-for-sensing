# path-prototype-hist-beam-adaptation Specification

## Purpose
定义 P3/HiST-Beam path prototype adaptation 的数据、输入边界、source-only prototype 构建、target 防泄漏和 LOSO 评估契约，用于审计 path-level propagation auxiliary 信号如何参与跨场景 few-shot 适配而不成为 sensing input。
## Requirements
### Requirement: P3/HiST path prototype 已退役
P3/HiST-Beam path prototype adaptation MUST 从当前支持面退役。系统 MUST 不再提供 P3-HiST 模型 forward、path prototype target adaptation、P3 smoke 配置或 P3 inference artifact。

#### Scenario: P3 Hist 配置不可运行
- **WHEN** 用户引用 P3/HiST path prototype 配置或 variant
- **THEN** 系统 MUST 报告该入口已退役或配置不存在
- **AND** 系统 MUST 不构建 HiST-Beam path prototype 模型

