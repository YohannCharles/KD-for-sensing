## ADDED Requirements

### Requirement: Data/training runtime wave 必须分层验证
Hotspot governance MUST 要求 data/training runtime 重构在 full regression 前运行 focused dataset、evaluation、objective 和 architecture tests。

#### Scenario: wave 验证命令记录
- **WHEN** a data/training runtime refactor wave completes
- **THEN** tasks or final implementation notes MUST list focused commands using `conda run -n kd_mm_beam`
- **AND** skipped commands MUST include reason and residual risk
