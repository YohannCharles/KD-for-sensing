## ADDED Requirements

### Requirement: AMBER-lite 与 AMBER full scope 分离
系统 MUST 区分 AMBER-lite local baseline 和 AMBER full architecture reproduction。AMBER-lite MUST 保持轻量缺失模态 baseline 语义；AMBER full MUST 使用独立配置、metadata、输出目录和 claim status。

#### Scenario: lite 配置保持 local lite scope
- **WHEN** 用户加载 `configs/fusion/amber_lite_missing_modality.yaml`
- **THEN** metadata MUST 继续记录 `reproduction_scope: amber_lite_local`
- **AND** 系统 MUST NOT 将该配置标记为完整 AMBER architecture reproduction

#### Scenario: full 配置使用 full scope
- **WHEN** 用户加载 AMBER full architecture 配置
- **THEN** metadata MUST 记录 `reproduction_scope: amber_full_local`
- **AND** 输出目录 MUST 与 AMBER-lite 输出目录区分
