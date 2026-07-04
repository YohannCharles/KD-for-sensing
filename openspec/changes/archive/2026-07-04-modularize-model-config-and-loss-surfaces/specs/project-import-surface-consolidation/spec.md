## ADDED Requirements

### Requirement: 模型配置拆分不得新增 barrel 或兼容 wrapper
Model/config/loss 重构 MUST 使用具体 owner import，并 MUST NOT 引入 package-level barrel、compatibility wrapper module、old registry alias facade 或跨领域 `utils` 模块来隐藏已移动 helper。

#### Scenario: helper 移动后调用方直连 owner
- **WHEN** helper code is moved out of a large model, loss or config module
- **THEN** internal callers MUST import the concrete owner module
- **AND** no new wrapper may exist solely to preserve an old private helper path
