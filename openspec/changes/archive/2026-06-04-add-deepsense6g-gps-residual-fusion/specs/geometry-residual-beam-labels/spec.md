## ADDED Requirements

### Requirement: GPS anchored signed circular residual utilities
系统 MUST 提供 GPS anchored residual correction 所需的 signed circular residual、beam shift、circular window 和 GPS good/bad label 工具，且 MUST 支持 torch 与 numpy 输入。

#### Scenario: signed circular residual wrap-around
- **WHEN** `target=1`、`pred=63`、`num_beams=64`
- **THEN** `signed_circular_residual` MUST 返回 `2`
- **AND** 当 `target=63`、`pred=1`、`num_beams=64` 时 MUST 返回 `-2`

#### Scenario: circular shift beam
- **WHEN** `pred=63`、`delta=2`、`num_beams=64`
- **THEN** `circular_shift_beam` MUST 返回 `1`
- **AND** 返回值 MUST 始终位于 `[0, num_beams)` 范围内

#### Scenario: circular window 处理边界
- **WHEN** `center=0`、`radius=2`、`num_beams=64`
- **THEN** `circular_window` MUST 包含 `62`、`63`、`0`、`1` 和 `2`
- **AND** window MUST 不包含重复 beam id

#### Scenario: GPS good bad label
- **WHEN** circular error 小于配置阈值
- **THEN** good label MUST 为 true 且 bad label MUST 为 false
- **AND** 阈值默认 MUST 为 `4`
- **AND** 该 label MUST 只用于 correction gate 训练和诊断，不得替代最终 DBA 指标
