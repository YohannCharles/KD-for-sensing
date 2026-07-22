## ADDED Requirements

### Requirement: PR-SQDF quick-validation 必须可追溯到冻结 T2/C0 owner
系统 MUST 将 claim-ineligible PR-SQDF cached-quality快筛视为 active T2研究任务，其 backbone、prototype、topology、MMW split和corruption generator MUST 追溯到 current C0/T2 owner。它 MUST 不新增 canonical recipe、whole-model family或兼容入口。

#### Scenario: 干净 clone 审计 current surface
- **WHEN** 维护者检查 PR-SQDF源码
- **THEN** package组件 MUST 不读取本地 C0 checkpoint或outputs目录
- **AND** 本地 checkpoint/config/cache路径 MUST 只由显式 analysis/launcher参数提供
