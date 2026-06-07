# radio-semantic-hist-beam-adaptation Specification

## Purpose
定义 radio-semantic HiST-Beam 适配的标签构造、dataset contract、模型输出、prototype artifact、target adaptation 和评估报告契约，使 radio 语义作为可审计的辅助监督与 prototype 对齐信号，而不是隐式改变 sensing 输入模态边界。
## Requirements
### Requirement: Radio-semantic Hist 已退役
Radio-semantic HiST-Beam adaptation MUST 从当前支持面退役。系统 MUST 不再提供 radio-semantic HiST model output、radio prototype artifact、radio-conditioned beam head、target adaptation 或 variant matrix。

#### Scenario: Radio-semantic Hist variant 不可构建
- **WHEN** 用户选择 radio-semantic HiST variant
- **THEN** 系统 MUST 报告该入口已退役或注册名不存在
- **AND** 系统 MUST 不构建 radio-conditioned HiST beam head

