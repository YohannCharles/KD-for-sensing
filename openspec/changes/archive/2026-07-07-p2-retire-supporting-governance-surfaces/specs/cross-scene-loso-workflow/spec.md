## ADDED Requirements

### Requirement: LOSO helper 退役必须先完成 current consumer 审计
`kd_sensing.data.loso` MAY 退役或删除，但 implementation MUST 先审计 OpenSpec current specs、docs、tests、scripts、configs 和 package imports。若仍存在 current consumer，helper MUST 保留并记录 retained-with-reason；若无 current consumer，`cross-scene-loso-workflow` MUST 降级为 historical/future contract 或同步移除对该 helper 的 current requirement。

#### Scenario: 无 current consumer 时退役 LOSO helper
- **WHEN** LOSO helper 没有 current docs/spec/tests/config/script/package consumer
- **THEN** implementation MAY 删除 `kd_sensing.data.loso`
- **AND** `cross-scene-loso-workflow` MUST 不再要求该 helper 作为 current source surface

#### Scenario: 仍有 current consumer 时保留 LOSO helper
- **WHEN** 任一 current workflow、test、config 或 documented command 仍调用 LOSO helper
- **THEN** implementation MUST 保留该 helper
- **AND** inventory MUST 记录 owner、当前消费者和未来删除触发条件

### Requirement: LOSO 退役不得破坏 split 可追溯性
若 LOSO helper 删除或降级，项目 MUST 保留已发布或已声明 cross-scene split 的可追溯说明，避免历史实验无法解释。

#### Scenario: historical split 仍可解释
- **WHEN** docs 或 claim notes 提到历史 LOSO/cross-scene split
- **THEN** 它们 MUST 指向 retained artifact、spec archive 或 documented split description
- **AND** MUST 不要求已删除 runtime helper 重新存在
