## MODIFIED Requirements

### Requirement: MMW evidence scripts 是 local/manual helper

保留的 all-weather、screening、BPA/CMA、BCACL inner-only 与 summary scripts MUST 有 MMW 或受限 DeepSense6G owner 和 output 边界，但 MUST 不注册为额外 console script。BCACL launcher MUST 只产生 single-seed、validation-only、claim-ineligible 的本地产物。

#### Scenario: 运行本地 helper

- **WHEN** 维护者运行 retained script
- **THEN** 其生成物 MUST 写入 local output root
- **AND** script MUST 不恢复历史 CLI 或 thin alias

#### Scenario: 启动 BCACL inner-only helper

- **WHEN** 维护者运行 BCACL launcher 或 validation evaluator
- **THEN** launcher MUST 显式关闭训练 outer test，evaluator MUST 只构造 validation split
- **AND** 两者不得注册为 package console script 或修改正式 claim
