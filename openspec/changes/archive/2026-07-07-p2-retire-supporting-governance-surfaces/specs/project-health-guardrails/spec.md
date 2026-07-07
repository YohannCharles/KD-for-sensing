## ADDED Requirements

### Requirement: Surface doctor 默认输出必须 issue-only
Project surface doctor 的默认输出 MUST 以问题、摘要和可执行 next action 为主。完整 pass inventory、allowlist dump、machine-readable governance table 或大段无问题清单 MUST 通过显式 opt-in flag 请求。

#### Scenario: 无问题时输出短摘要
- **WHEN** project surface doctor 运行且没有发现问题
- **THEN** 默认输出 MUST 是短摘要，说明检查 scope 和无问题状态
- **AND** MUST 不默认打印完整 pass inventory JSON 或大段逐项清单

#### Scenario: 完整清单显式 opt-in
- **WHEN** 协作者需要完整 inventory、allowlist 或 machine-readable dump
- **THEN** MUST 使用 `--dump-inventory` 或等价显式 flag
- **AND** 输出格式 MUST 在 help text 中说明适合审计/机器处理，而不是默认人读路径

### Requirement: Surface doctor 瘦身不得降低失败可诊断性
默认输出变短后，任何失败或 warning MUST 仍包含 scope、问题路径、原因和建议 next action。

#### Scenario: 有问题时保留 actionable detail
- **WHEN** surface doctor 检查发现 missing、stale、orphan、complete-unarchived 或 wrapper 回流问题
- **THEN** 默认输出 MUST 包含足够定位和修复的信息
- **AND** full dump flag MAY 提供额外上下文但不能成为理解失败的必要条件
